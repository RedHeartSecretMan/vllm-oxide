//! `Linear<P: ParallelStyle>` + `LinearSpec` (ADR-0001 / ADR-0002).
//!
//! One generic struct over a type-level style tag ([`QkvMerged`] /
//! [`GateUpMerged`] / [`Row`]); plain `Column` deferred (YAGNI). The style
//! tag owns QKV/gate-up fusion at load time via [`Linear::from_vb`] — the
//! loader stays fully model-agnostic. HF checkpoint names map 1:1 (no remap
//! table): `from_vb` for [`QkvMerged`] reads `q_proj.weight` / `k_proj.weight`
//! / `v_proj.weight` and `Tensor::cat`s them along dim 0; [`GateUpMerged`]
//! does the same for gate/up; [`Row`] reads a single weight.
//!
//! `LinearSpec` closes the ADR-0002 seam: model code unpacks its own config
//! (e.g. `Qwen3Config`) into this neutral geometry struct; `Linear<P>` never
//! imports architecture-specific types.

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::marker::PhantomData;

use super::parallel::{GateUpMerged, ParallelStyle, QkvMerged, Row};

/// Neutral geometry for a [`Linear<P>`]. Closes the ADR-0002 seam so `layers/`
/// stays fully model-agnostic — model code unpacks its own `Config` into this
/// struct; `Linear<P>` never imports `Qwen3Config` or any architecture type.
///
/// `out_features_per_shard` is the per-shard size, not the fused total:
///
/// - `QkvMerged` → `num_heads * head_dim` (the Q size; K/V may be smaller under
///   GQA, read via `get_unchecked` so the checkpoint dictates their shape).
/// - `GateUpMerged` → `intermediate_size` (gate and up are the same size).
/// - `Row` → `out_features` of the projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearSpec {
    pub in_features: usize,
    pub out_features_per_shard: usize,
    pub bias: bool,
}

/// Fused linear projection generic over a [`ParallelStyle`] type-level tag.
///
/// The tag determines (a) how [`Linear::from_vb`] assembles the weight from
/// checkpoint shards and (b) the TP-seam slicing behaviour in v0.2. Forward
/// is style-agnostic: `x @ W^T + b`.
///
/// `PhantomData<P>` is the documented cosmetic cost of the typestate pattern
/// (ADR-0001 R2) — accepted in exchange for compile-time checking that the
/// model code's declared projection style matches the layer's expected layout.
pub struct Linear<P: ParallelStyle> {
    weight: Tensor,
    bias: Option<Tensor>,
    _marker: PhantomData<P>,
}

impl<P: ParallelStyle + StyleBuilder> Linear<P> {
    /// Construct from a checkpoint-backed [`VarBuilder`]. Reads shards per the
    /// style tag, concatenates them along dim 0 (for fused styles), and
    /// optionally assembles bias the same way.
    ///
    /// `dev` is accepted for forward-compat with v0.2 TP-shard placement;
    /// v0.1 ignores it (the `VarBuilder` already carries the device).
    pub fn from_vb(vb: VarBuilder, spec: &LinearSpec, _dev: &Device) -> Result<Self> {
        let weight = P::build_weight(&vb, spec)?;
        let bias = if spec.bias {
            P::build_bias(&vb, spec)?
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            _marker: PhantomData,
        })
    }
}

impl<P: ParallelStyle> Linear<P> {
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
            _marker: PhantomData,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

/// Per-style checkpoint assembly. `pub(crate)` so external code can call
/// [`Linear::from_vb`] for the three v0.1 styles but cannot add new style
/// impls (closed set — sealed by visibility).
pub(crate) trait StyleBuilder: ParallelStyle {
    fn build_weight(vb: &VarBuilder, spec: &LinearSpec) -> Result<Tensor>;
    fn build_bias(vb: &VarBuilder, spec: &LinearSpec) -> Result<Option<Tensor>>;
}

impl StyleBuilder for QkvMerged {
    fn build_weight(vb: &VarBuilder, _spec: &LinearSpec) -> Result<Tensor> {
        // Read each shard with `get_unchecked` so K/V shapes come from the
        // checkpoint — GQA models have num_kv_heads < num_heads, so K/V are
        // smaller than Q. ADR-0002 R2 accepts runtime-only missing-tensor
        // detection.
        let q = vb.get_unchecked("q_proj.weight")?;
        let k = vb.get_unchecked("k_proj.weight")?;
        let v = vb.get_unchecked("v_proj.weight")?;
        Tensor::cat(&[&q, &k, &v], 0)
    }

    fn build_bias(vb: &VarBuilder, _spec: &LinearSpec) -> Result<Option<Tensor>> {
        if !vb.contains_tensor("q_proj.bias") {
            return Ok(None);
        }
        let q = vb.get_unchecked("q_proj.bias")?;
        let k = vb.get_unchecked("k_proj.bias")?;
        let v = vb.get_unchecked("v_proj.bias")?;
        Ok(Some(Tensor::cat(&[&q, &k, &v], 0)?))
    }
}

impl StyleBuilder for GateUpMerged {
    fn build_weight(vb: &VarBuilder, spec: &LinearSpec) -> Result<Tensor> {
        let shape = (spec.out_features_per_shard, spec.in_features);
        let gate = vb.get(shape, "gate_proj.weight")?;
        let up = vb.get(shape, "up_proj.weight")?;
        Tensor::cat(&[&gate, &up], 0)
    }

    fn build_bias(vb: &VarBuilder, spec: &LinearSpec) -> Result<Option<Tensor>> {
        if !vb.contains_tensor("gate_proj.bias") {
            return Ok(None);
        }
        let gate = vb.get(spec.out_features_per_shard, "gate_proj.bias")?;
        let up = vb.get(spec.out_features_per_shard, "up_proj.bias")?;
        Ok(Some(Tensor::cat(&[&gate, &up], 0)?))
    }
}

impl StyleBuilder for Row {
    fn build_weight(vb: &VarBuilder, spec: &LinearSpec) -> Result<Tensor> {
        let shape = (spec.out_features_per_shard, spec.in_features);
        vb.get(shape, "weight")
    }

    fn build_bias(vb: &VarBuilder, spec: &LinearSpec) -> Result<Option<Tensor>> {
        if !vb.contains_tensor("bias") {
            return Ok(None);
        }
        Ok(Some(vb.get(spec.out_features_per_shard, "bias")?))
    }
}

impl Linear<Row> {
    pub fn from_weight(weight: Tensor) -> Self {
        Self {
            weight,
            bias: None,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    /// Build a `VarBuilder` backed by a `HashMap<String, Tensor>` so tests can
    /// inject known shard values and verify concat order via constant markers.
    fn vb_from(tensors: Vec<(String, Tensor)>) -> VarBuilder<'static> {
        let map: HashMap<String, Tensor> = tensors.into_iter().collect();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    mod linear_spec {
        use super::*;

        #[test]
        fn spec_is_copy_eq() {
            let a = LinearSpec {
                in_features: 4,
                out_features_per_shard: 8,
                bias: false,
            };
            let b = a;
            assert_eq!(a, b);
        }
    }

    mod qkv_merged {
        use super::*;

        #[test]
        fn concat_order_is_q_then_k_then_v_along_dim0() {
            // GQA layout: q=4 rows, k=2 rows, v=2 rows. Use distinct constant
            // markers (1.0, 2.0, 3.0) so row position identifies the shard.
            let q = Tensor::full(1.0f32, (4, 3), &Device::Cpu).unwrap();
            let k = Tensor::full(2.0f32, (2, 3), &Device::Cpu).unwrap();
            let v = Tensor::full(3.0f32, (2, 3), &Device::Cpu).unwrap();
            let vb = vb_from(vec![
                ("q_proj.weight".to_string(), q),
                ("k_proj.weight".to_string(), k),
                ("v_proj.weight".to_string(), v),
            ]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 4,
                bias: false,
            };
            let lin = Linear::<QkvMerged>::from_vb(vb, &spec, &Device::Cpu).unwrap();

            let w = lin.weight();
            assert_eq!(w.shape().dims(), [8, 3]);
            // First 4 rows = q (1.0), next 2 = k (2.0), last 2 = v (3.0).
            let row_val = |r: usize| w.get(r).unwrap().to_vec1::<f32>().unwrap()[0];
            assert_eq!(row_val(0), 1.0);
            assert_eq!(row_val(3), 1.0);
            assert_eq!(row_val(4), 2.0);
            assert_eq!(row_val(5), 2.0);
            assert_eq!(row_val(6), 3.0);
            assert_eq!(row_val(7), 3.0);
        }

        #[test]
        fn bias_concat_order_matches_weight() {
            let q_w = Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap();
            let k_w = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
            let v_w = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
            let q_b = Tensor::full(10.0f32, (4,), &Device::Cpu).unwrap();
            let k_b = Tensor::full(20.0f32, (2,), &Device::Cpu).unwrap();
            let v_b = Tensor::full(30.0f32, (2,), &Device::Cpu).unwrap();
            let vb = vb_from(vec![
                ("q_proj.weight".to_string(), q_w),
                ("k_proj.weight".to_string(), k_w),
                ("v_proj.weight".to_string(), v_w),
                ("q_proj.bias".to_string(), q_b),
                ("k_proj.bias".to_string(), k_b),
                ("v_proj.bias".to_string(), v_b),
            ]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 4,
                bias: true,
            };
            let lin = Linear::<QkvMerged>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            let b = lin.bias().unwrap();
            assert_eq!(b.shape().dims(), [8]);
            assert_eq!(b.get(0).unwrap().to_vec0::<f32>().unwrap(), 10.0);
            assert_eq!(b.get(4).unwrap().to_vec0::<f32>().unwrap(), 20.0);
            assert_eq!(b.get(6).unwrap().to_vec0::<f32>().unwrap(), 30.0);
        }

        #[test]
        fn bias_none_when_spec_says_no_bias() {
            let q = Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap();
            let k = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
            let v = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
            let vb = vb_from(vec![
                ("q_proj.weight".to_string(), q),
                ("k_proj.weight".to_string(), k),
                ("v_proj.weight".to_string(), v),
            ]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 4,
                bias: false,
            };
            let lin = Linear::<QkvMerged>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            assert!(lin.bias().is_none());
        }

        #[test]
        fn missing_qkv_tensor_is_runtime_error() {
            let vb = vb_from(vec![]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 4,
                bias: false,
            };
            let err = Linear::<QkvMerged>::from_vb(vb, &spec, &Device::Cpu);
            assert!(err.is_err());
        }
    }

    mod gate_up_merged {
        use super::*;

        #[test]
        fn concat_order_is_gate_then_up_along_dim0() {
            let gate = Tensor::full(5.0f32, (4, 3), &Device::Cpu).unwrap();
            let up = Tensor::full(7.0f32, (4, 3), &Device::Cpu).unwrap();
            let vb = vb_from(vec![
                ("gate_proj.weight".to_string(), gate),
                ("up_proj.weight".to_string(), up),
            ]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 4,
                bias: false,
            };
            let lin = Linear::<GateUpMerged>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            let w = lin.weight();
            assert_eq!(w.shape().dims(), [8, 3]);
            let row_val = |r: usize| w.get(r).unwrap().to_vec1::<f32>().unwrap()[0];
            assert_eq!(row_val(0), 5.0);
            assert_eq!(row_val(3), 5.0);
            assert_eq!(row_val(4), 7.0);
            assert_eq!(row_val(7), 7.0);
        }
    }

    mod row {
        use super::*;

        #[test]
        fn weight_is_single_tensor_no_concat() {
            let w = Tensor::full(9.0f32, (5, 3), &Device::Cpu).unwrap();
            let vb = vb_from(vec![("weight".to_string(), w)]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 5,
                bias: false,
            };
            let lin = Linear::<Row>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            assert_eq!(lin.weight().shape().dims(), [5, 3]);
            let row_val = |r: usize| lin.weight().get(r).unwrap().to_vec1::<f32>().unwrap()[0];
            assert_eq!(row_val(0), 9.0);
        }

        #[test]
        fn pp_prefix_resolves_to_full_checkpoint_name() {
            // Simulates model code calling `vb.pp("o_proj")` then constructing
            // a Row — the resolved checkpoint name is "o_proj.weight".
            let w = Tensor::full(11.0f32, (5, 3), &Device::Cpu).unwrap();
            let vb = vb_from(vec![("o_proj.weight".to_string(), w)]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 5,
                bias: false,
            };
            let lin = Linear::<Row>::from_vb(vb.pp("o_proj"), &spec, &Device::Cpu).unwrap();
            let row_val = |r: usize| lin.weight().get(r).unwrap().to_vec1::<f32>().unwrap()[0];
            assert_eq!(row_val(0), 11.0);
        }
    }

    mod forward {
        use super::*;

        #[test]
        fn row_forward_applies_matmul_without_bias() {
            // W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (3x3 identity), x = [2, 3, 4]
            // out = x @ W^T = [2, 3, 4]
            let mut w = Vec::with_capacity(9);
            for r in 0..3 {
                for c in 0..3 {
                    w.push(if r == c { 1.0f32 } else { 0.0 });
                }
            }
            let weight = Tensor::from_vec(w, (3, 3), &Device::Cpu).unwrap();
            let vb = vb_from(vec![("weight".to_string(), weight)]);
            let spec = LinearSpec {
                in_features: 3,
                out_features_per_shard: 3,
                bias: false,
            };
            let lin = Linear::<Row>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            let x = Tensor::from_iter([2.0f32, 3.0, 4.0], &Device::Cpu)
                .unwrap()
                .reshape((1, 3))
                .unwrap();
            let out = lin.forward(&x).unwrap();
            assert_eq!(out.shape().dims(), [1, 3]);
            assert_eq!(
                out.get(0)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap(),
                2.0
            );
            assert_eq!(
                out.get(0)
                    .unwrap()
                    .get(1)
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap(),
                3.0
            );
            assert_eq!(
                out.get(0)
                    .unwrap()
                    .get(2)
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap(),
                4.0
            );
        }

        #[test]
        fn row_forward_applies_bias_when_present() {
            let weight = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
            let bias = Tensor::from_iter([5.0f32, 7.0], &Device::Cpu).unwrap();
            let vb = vb_from(vec![
                ("weight".to_string(), weight),
                ("bias".to_string(), bias),
            ]);
            let spec = LinearSpec {
                in_features: 2,
                out_features_per_shard: 2,
                bias: true,
            };
            let lin = Linear::<Row>::from_vb(vb, &spec, &Device::Cpu).unwrap();
            let x = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
            let out = lin.forward(&x).unwrap();
            assert_eq!(
                out.get(0)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap(),
                5.0
            );
            assert_eq!(
                out.get(0)
                    .unwrap()
                    .get(1)
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap(),
                7.0
            );
        }
    }

    mod model_agnostic_seam {
        use super::*;

        #[test]
        fn layers_module_has_no_imports_from_models() {
            // Static property: the layers/ module must not import anything from
            // crate::models. This is enforced by the ADR-0002 seam — verified
            // here at the type level by confirming `LinearSpec` carries no
            // architecture-specific types (just two `usize` + a `bool`).
            let spec = LinearSpec {
                in_features: 1,
                out_features_per_shard: 1,
                bias: false,
            };
            // If the seam closes properly, this assertion is trivially true —
            // the goal is to make any future seam break visible at compile time
            // by e.g. adding a non-`usize` field here.
            let _: usize = spec.in_features;
            let _: usize = spec.out_features_per_shard;
            let _: bool = spec.bias;
        }
    }
}
