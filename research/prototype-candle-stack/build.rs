// Throwaway prototype build.rs — mirrors mistral.rs's mistralrs-paged-attn/build.rs
// pattern (lines 50-148) but stripped to the minimum needed to compile one .cu file.
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Rebuild if any kernel source changes.
    println!("cargo:rerun-if-changed=kernels/fill_const.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let lib = out_dir.join("libfillconst.a");

    // cudaforge::KernelBuilder is what mistral.rs and candle-flash-attn both use.
    // It auto-detects compute capability and emits the right nvcc gencode.
    cudaforge::KernelBuilder::new()
        .source_glob("kernels/*.cu")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--compiler-options")
        .arg("-fPIC")
        .build_lib(lib)
        .map_err(|e| anyhow::anyhow!("cudaforge build failed: {e}"))?;

    // The ONLY link wiring — no #[link] attrs in Rust source (mirrors mistral.rs).
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=fillconst");
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}
