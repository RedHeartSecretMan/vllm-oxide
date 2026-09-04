//! Resolve one model source into an immutable identity and artifact set.
//!
//! [`ResolvedModel`] is the construction seam: configuration bytes, tokenizer,
//! special-token metadata, weight shards, and dtype all flow through this one
//! value. A Hub branch or tag is queried once for its commit, after which every
//! artifact is fetched by that exact commit. Offline mode resolves the cached
//! ref to one `snapshots/<commit>` directory before reading artifact contents.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::DType;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::config::{default_dtype_from_config_json, Source};

/// Stable identity shared by every artifact consumed during one construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelIdentity {
    /// One canonical local directory used for every artifact lookup.
    Local { root: PathBuf },
    /// One Hub repository pinned to an immutable commit.
    Hub { repo: String, commit: String },
}

impl std::fmt::Display for ModelIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local { root } => write!(formatter, "local:{}", root.display()),
            Self::Hub { repo, commit } => write!(formatter, "hub:{repo}@{commit}"),
        }
    }
}

/// Special-token metadata required by the generation engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpecialTokenIds {
    eos: Vec<u32>,
    bos: Option<u32>,
    pad: Option<u32>,
}

impl SpecialTokenIds {
    pub(crate) fn eos(&self) -> &[u32] {
        &self.eos
    }

    pub(crate) fn bos(&self) -> Option<u32> {
        self.bos
    }

    pub(crate) fn pad(&self) -> Option<u32> {
        self.pad
    }
}

#[derive(Debug, Clone)]
struct HubRevision {
    commit: String,
    files: BTreeSet<String>,
}

trait HubAccess {
    fn resolve_revision(&mut self, repo: &str, revision: &str) -> Result<HubRevision>;

    fn get(&mut self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf>;
}

struct OnlineHubAccess {
    api: hf_hub::api::sync::Api,
}

impl OnlineHubAccess {
    fn new() -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_progress(true)
            .build()
            .context("building hf-hub sync client")?;
        Ok(Self { api })
    }
}

impl HubAccess for OnlineHubAccess {
    fn resolve_revision(&mut self, repo: &str, revision: &str) -> Result<HubRevision> {
        let info = self
            .api
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .info()
            .with_context(|| format!("querying Hub metadata for `{repo}` at `{revision}`"))?;
        if info.sha.trim().is_empty() {
            bail!("Hub returned an empty commit for `{repo}` at `{revision}`");
        }
        Ok(HubRevision {
            commit: info.sha,
            files: info
                .siblings
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .collect(),
        })
    }

    fn get(&mut self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf> {
        self.api
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                commit.to_string(),
            ))
            .get(filename)
            .with_context(|| format!("downloading `{filename}` from `{repo}` at commit `{commit}`"))
    }
}

struct OfflineHubAccess {
    cache: hf_hub::Cache,
}

impl OfflineHubAccess {
    fn new(cache: hf_hub::Cache) -> Self {
        Self { cache }
    }
}

impl HubAccess for OfflineHubAccess {
    fn resolve_revision(&mut self, repo: &str, revision: &str) -> Result<HubRevision> {
        let cache_repo = self.cache.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        let config_path = cache_repo.get("config.json").or_else(|| {
            let direct = cache_repo.pointer_path(revision).join("config.json");
            direct.is_file().then_some(direct)
        });
        let config_path = config_path.ok_or_else(|| {
            anyhow::anyhow!(
                "HF_HUB_OFFLINE=1 and no cached config.json for `{repo}` revision `{revision}`"
            )
        })?;
        let snapshot_root = config_path.parent().ok_or_else(|| {
            anyhow::anyhow!("cached config path {} has no parent", config_path.display())
        })?;
        let commit = snapshot_root
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot derive immutable commit from cached snapshot {}",
                    snapshot_root.display()
                )
            })?
            .to_string();
        let files = collect_snapshot_files(snapshot_root)?;
        Ok(HubRevision { commit, files })
    }

    fn get(&mut self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf> {
        let path = self
            .cache
            .repo(hf_hub::Repo::with_revision(
                repo.to_string(),
                hf_hub::RepoType::Model,
                commit.to_string(),
            ))
            .pointer_path(commit)
            .join(filename);
        if !path.is_file() {
            bail!(
                "cached Hub model `{repo}` at commit `{commit}` has no `{filename}` at {}",
                path.display()
            );
        }
        Ok(path)
    }
}

/// A source resolved once into an immutable identity and concrete artifacts.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedModel {
    identity: ModelIdentity,
    config_path: PathBuf,
    config_json: Vec<u8>,
    generation_config_json: Option<Vec<u8>>,
    tokenizer_path: PathBuf,
    weight_paths: Vec<PathBuf>,
    dtype: DType,
}

impl ResolvedModel {
    pub(crate) fn resolve(source: Source, requested_dtype: Option<DType>) -> Result<Self> {
        match source {
            Source::Local(root) => Self::resolve_local(root, requested_dtype),
            Source::Hub { repo, revision } => {
                if crate::config::is_hf_hub_offline() {
                    Self::resolve_cached_hub(
                        repo,
                        revision.unwrap_or_else(|| "main".to_string()),
                        requested_dtype,
                        hf_hub::Cache::from_env(),
                    )
                } else {
                    let mut hub = OnlineHubAccess::new()?;
                    Self::resolve_with_hub(
                        Source::Hub { repo, revision },
                        requested_dtype,
                        &mut hub,
                    )
                }
            }
        }
    }

    fn resolve_cached_hub(
        repo: String,
        revision: String,
        requested_dtype: Option<DType>,
        cache: hf_hub::Cache,
    ) -> Result<Self> {
        let mut hub = OfflineHubAccess::new(cache);
        Self::resolve_with_hub(
            Source::Hub {
                repo,
                revision: Some(revision),
            },
            requested_dtype,
            &mut hub,
        )
    }

    fn resolve_with_hub(
        source: Source,
        requested_dtype: Option<DType>,
        hub: &mut dyn HubAccess,
    ) -> Result<Self> {
        validate_requested_dtype(requested_dtype)
            .with_context(|| format!("validating dtype for model source {source:?}"))?;
        let Source::Hub { repo, revision } = source else {
            return Self::resolve(source, requested_dtype);
        };
        let requested_revision = revision.as_deref().unwrap_or("main");
        let resolved = hub
            .resolve_revision(&repo, requested_revision)
            .with_context(|| {
                format!("resolving Hub model `{repo}` revision `{requested_revision}`")
            })?;
        for required in ["config.json", "tokenizer.json"] {
            if !resolved.files.contains(required) {
                bail!(
                    "resolved Hub model `{repo}` at commit `{}` has no `{required}`",
                    resolved.commit
                );
            }
        }

        let config_path = hub
            .get(&repo, &resolved.commit, "config.json")
            .with_context(|| {
                format!(
                    "fetching config.json for Hub model `{repo}` at commit `{}`",
                    resolved.commit
                )
            })?;
        let tokenizer_path = hub
            .get(&repo, &resolved.commit, "tokenizer.json")
            .with_context(|| {
                format!(
                    "fetching tokenizer.json for Hub model `{repo}` at commit `{}`",
                    resolved.commit
                )
            })?;
        let generation_config_json =
            if resolved.files.contains("generation_config.json") {
                let path = hub
                    .get(&repo, &resolved.commit, "generation_config.json")
                    .with_context(|| {
                        format!(
                            "fetching generation_config.json for Hub model `{repo}` at commit `{}`",
                            resolved.commit
                        )
                    })?;
                Some(std::fs::read(&path).with_context(|| {
                    format!("reading generation metadata from {}", path.display())
                })?)
            } else {
                None
            };
        let weight_paths = if resolved.files.contains("model.safetensors.index.json") {
            let index_path = hub
                .get(&repo, &resolved.commit, "model.safetensors.index.json")
                .with_context(|| {
                    format!(
                        "fetching safetensors index for Hub model `{repo}` at commit `{}`",
                        resolved.commit
                    )
                })?;
            let shard_names = parse_index_shard_names(&index_path)?;
            let mut paths = Vec::with_capacity(shard_names.len());
            for shard in shard_names {
                if !resolved.files.contains(&shard) {
                    bail!(
                        "safetensors index for Hub model `{repo}` at commit `{}` references missing shard `{shard}`",
                        resolved.commit
                    );
                }
                paths.push(hub.get(&repo, &resolved.commit, &shard).with_context(|| {
                    format!(
                        "fetching shard `{shard}` for Hub model `{repo}` at commit `{}`",
                        resolved.commit
                    )
                })?);
            }
            paths
        } else if resolved.files.contains("model.safetensors") {
            vec![hub
                .get(&repo, &resolved.commit, "model.safetensors")
                .with_context(|| {
                    format!(
                        "fetching model.safetensors for Hub model `{repo}` at commit `{}`",
                        resolved.commit
                    )
                })?]
        } else {
            bail!(
                "resolved Hub model `{repo}` at commit `{}` has neither `model.safetensors.index.json` nor `model.safetensors`",
                resolved.commit
            );
        };
        let config_json = std::fs::read(&config_path).with_context(|| {
            format!(
                "reading config.json for Hub model `{repo}` at commit `{}` from {}",
                resolved.commit,
                config_path.display()
            )
        })?;
        let dtype = select_dtype(&config_json, requested_dtype).with_context(|| {
            format!(
                "resolving dtype for Hub model `{repo}` at commit `{}`",
                resolved.commit
            )
        })?;

        Ok(Self {
            identity: ModelIdentity::Hub {
                repo,
                commit: resolved.commit,
            },
            config_path,
            config_json,
            generation_config_json,
            tokenizer_path,
            weight_paths,
            dtype,
        })
    }

    fn resolve_local(root: PathBuf, requested_dtype: Option<DType>) -> Result<Self> {
        validate_requested_dtype(requested_dtype)
            .with_context(|| format!("validating dtype for local model {}", root.display()))?;
        let root = root
            .canonicalize()
            .with_context(|| format!("resolving local model root {}", root.display()))?;
        let config_path = root.join("config.json");
        let config_json = std::fs::read(&config_path)
            .with_context(|| format!("reading config.json from {}", config_path.display()))?;
        let tokenizer_path = root.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            bail!("tokenizer.json not found at {}", tokenizer_path.display());
        }
        let generation_config_json = read_optional_file(&root.join("generation_config.json"))?;
        let weight_paths = resolve_local_weight_paths(&root)?;
        let dtype = select_dtype(&config_json, requested_dtype)
            .with_context(|| format!("resolving dtype for local model {}", root.display()))?;

        Ok(Self {
            identity: ModelIdentity::Local { root },
            config_path,
            config_json,
            generation_config_json,
            tokenizer_path,
            weight_paths,
            dtype,
        })
    }

    pub(crate) fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    pub(crate) fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub(crate) fn config_json(&self) -> &[u8] {
        &self.config_json
    }

    #[cfg(test)]
    pub(crate) fn generation_config_json(&self) -> Option<&[u8]> {
        self.generation_config_json.as_deref()
    }

    pub(crate) fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub(crate) fn weight_paths(&self) -> &[PathBuf] {
        &self.weight_paths
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn load_tokenizer(&self) -> Result<Tokenizer> {
        Tokenizer::from_file(self.tokenizer_path()).map_err(|error| {
            anyhow::anyhow!(
                "loading tokenizer for model identity `{}` from {}: {error}",
                self.identity,
                self.tokenizer_path.display()
            )
        })
    }

    pub(crate) fn special_token_ids(&self, tokenizer: &Tokenizer) -> Result<SpecialTokenIds> {
        let model: SpecialTokenConfig =
            serde_json::from_slice(&self.config_json).with_context(|| {
                format!(
                    "parsing special-token metadata from config.json for `{}`",
                    self.identity
                )
            })?;
        let generation: Option<SpecialTokenConfig> = self
            .generation_config_json
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .with_context(|| {
                format!(
                    "parsing special-token metadata from generation_config.json for `{}`",
                    self.identity
                )
            })?;
        let generation = generation.unwrap_or_default();
        let eos = generation
            .eos_token_id
            .or(model.eos_token_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "model identity `{}` has no required `eos_token_id` metadata",
                    self.identity
                )
            })?
            .into_vec();
        let mut unique_eos = Vec::with_capacity(eos.len());
        for id in eos {
            if !unique_eos.contains(&id) {
                unique_eos.push(id);
            }
        }
        if unique_eos.is_empty() {
            bail!(
                "model identity `{}` has an empty `eos_token_id` list",
                self.identity
            );
        }
        let special = SpecialTokenIds {
            eos: unique_eos,
            bos: generation.bos_token_id.or(model.bos_token_id),
            pad: generation.pad_token_id.or(model.pad_token_id),
        };

        for (kind, id) in special
            .eos
            .iter()
            .copied()
            .map(|id| ("eos_token_id", id))
            .chain(special.bos.map(|id| ("bos_token_id", id)))
            .chain(special.pad.map(|id| ("pad_token_id", id)))
        {
            let token = tokenizer.id_to_token(id).ok_or_else(|| {
                anyhow::anyhow!(
                    "model identity `{}` declares {kind}={id}, but tokenizer.json has no token for that id",
                    self.identity
                )
            })?;
            if tokenizer.token_to_id(&token) != Some(id) {
                bail!(
                    "model identity `{}` has inconsistent tokenizer mapping for {kind}={id} (`{token}`)",
                    self.identity
                );
            }
        }

        Ok(special)
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct SpecialTokenConfig {
    #[serde(default)]
    eos_token_id: Option<OneOrManyTokenIds>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn parse_index_shard_names(index_path: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(index_path)
        .with_context(|| format!("reading safetensors index {}", index_path.display()))?;
    let parsed: SafetensorsIndex = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing safetensors index {}", index_path.display()))?;
    let names: BTreeSet<String> = parsed.weight_map.into_values().collect();
    if names.is_empty() {
        bail!(
            "safetensors index {} has an empty `weight_map`",
            index_path.display()
        );
    }
    for name in &names {
        let path = Path::new(name);
        if name.is_empty()
            || path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!(
                "safetensors shard `{name}` from {} points outside resolved model identity",
                index_path.display()
            );
        }
    }
    Ok(names.into_iter().collect())
}

fn resolve_local_weight_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let index_path = root.join("model.safetensors.index.json");
    if index_path.is_file() {
        let names = parse_index_shard_names(&index_path)?;
        let paths: Vec<PathBuf> = names.into_iter().map(|name| root.join(name)).collect();
        for path in &paths {
            if !path.is_file() {
                bail!(
                    "shard {} is referenced by {} but is not present",
                    path.display(),
                    index_path.display()
                );
            }
        }
        return Ok(paths);
    }

    let single = root.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(root)
        .with_context(|| format!("reading local model root {}", root.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!(
            "no safetensors weights under local model root {}",
            root.display()
        );
    }
    Ok(paths)
}

fn read_optional_file(path: &Path) -> Result<Option<Vec<u8>>> {
    if path.is_file() {
        Ok(Some(
            std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        ))
    } else {
        Ok(None)
    }
}

fn select_dtype(config_json: &[u8], requested: Option<DType>) -> Result<DType> {
    if let Some(dtype) = requested {
        return Ok(dtype);
    }
    let dtype = default_dtype_from_config_json(config_json)
        .context("reading model config `torch_dtype`")?;
    validate_supported_dtype(dtype, "model config dtype")
}

fn validate_requested_dtype(requested: Option<DType>) -> Result<()> {
    requested
        .map(|dtype| validate_supported_dtype(dtype, "requested dtype"))
        .transpose()
        .map(|_| ())
}

fn validate_supported_dtype(dtype: DType, origin: &str) -> Result<DType> {
    if matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
        Ok(dtype)
    } else {
        bail!(
            "{origin} {dtype:?} is unsupported; supported model and KV-cache dtypes: BF16, F16, F32"
        )
    }
}

fn collect_snapshot_files(root: &Path) -> Result<BTreeSet<String>> {
    let mut files = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("reading cached snapshot directory {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("reading entry under {}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.is_file() {
                let relative = path.strip_prefix(root).with_context(|| {
                    format!(
                        "cached artifact {} is outside snapshot {}",
                        path.display(),
                        root.display()
                    )
                })?;
                files.insert(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use anyhow::Result;
    use candle_core::DType;
    use std::collections::BTreeSet;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer};

    use super::*;
    use crate::config::Source;

    fn write_tokenizer(path: &Path) {
        let vocab_path = path.with_file_name("test-vocab.json");
        std::fs::write(
            &vocab_path,
            br#"{"<pad>":0,"<bos>":1,"<eos>":2,"<stop>":3,"<unk>":4}"#,
        )
        .unwrap();
        let model =
            WordLevel::from_file(vocab_path.to_str().unwrap(), "<unk>".to_string()).unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.add_special_tokens(&[
            AddedToken::from("<pad>", true),
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
            AddedToken::from("<stop>", true),
        ]);
        tokenizer.save(path, false).unwrap();
    }

    #[test]
    fn local_source_resolves_one_identity_for_every_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":7}"#,
        )
        .unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), b"").unwrap();

        let resolved =
            ResolvedModel::resolve(Source::Local(tmp.path().to_path_buf()), Some(DType::F16))
                .unwrap();
        let root = tmp.path().canonicalize().unwrap();

        assert_eq!(
            resolved.identity(),
            &ModelIdentity::Local { root: root.clone() }
        );
        assert_eq!(resolved.dtype(), DType::F16);
        assert_eq!(resolved.config_path(), root.join("config.json"));
        assert_eq!(resolved.tokenizer_path(), root.join("tokenizer.json"));
        assert_eq!(resolved.weight_paths(), &[root.join("model.safetensors")]);
    }

    #[derive(Debug)]
    struct FakeHub {
        snapshot_root: PathBuf,
        commit: String,
        files: BTreeSet<String>,
        resolve_calls: Vec<(String, String)>,
        get_calls: Vec<(String, String, String)>,
    }

    impl HubAccess for FakeHub {
        fn resolve_revision(&mut self, repo: &str, revision: &str) -> Result<HubRevision> {
            self.resolve_calls
                .push((repo.to_string(), revision.to_string()));
            Ok(HubRevision {
                commit: self.commit.clone(),
                files: self.files.clone(),
            })
        }

        fn get(&mut self, repo: &str, commit: &str, filename: &str) -> Result<PathBuf> {
            self.get_calls
                .push((repo.to_string(), commit.to_string(), filename.to_string()));
            Ok(self.snapshot_root.join(filename))
        }
    }

    #[test]
    fn remote_source_resolves_once_then_fetches_every_artifact_by_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let snapshot_root = tmp.path().join("snapshots").join(commit);
        std::fs::create_dir_all(&snapshot_root).unwrap();
        std::fs::write(
            snapshot_root.join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":7}"#,
        )
        .unwrap();
        std::fs::write(snapshot_root.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(snapshot_root.join("model.safetensors"), b"").unwrap();
        let mut hub = FakeHub {
            snapshot_root: snapshot_root.clone(),
            commit: commit.to_string(),
            files: BTreeSet::from([
                "config.json".to_string(),
                "model.safetensors".to_string(),
                "tokenizer.json".to_string(),
            ]),
            resolve_calls: Vec::new(),
            get_calls: Vec::new(),
        };

        let resolved = ResolvedModel::resolve_with_hub(
            Source::Hub {
                repo: "org/model".to_string(),
                revision: Some("moving-branch".to_string()),
            },
            None,
            &mut hub,
        )
        .unwrap();

        assert_eq!(
            resolved.identity(),
            &ModelIdentity::Hub {
                repo: "org/model".to_string(),
                commit: commit.to_string(),
            }
        );
        assert_eq!(
            hub.resolve_calls,
            vec![("org/model".to_string(), "moving-branch".to_string())]
        );
        assert_eq!(hub.get_calls.len(), 3);
        assert!(hub
            .get_calls
            .iter()
            .all(|(repo, revision, _)| repo == "org/model" && revision == commit));
        assert_eq!(resolved.config_path(), snapshot_root.join("config.json"));
        assert_eq!(
            resolved.tokenizer_path(),
            snapshot_root.join("tokenizer.json")
        );
        assert_eq!(
            resolved.weight_paths(),
            &[snapshot_root.join("model.safetensors")]
        );
    }

    #[test]
    fn indexed_remote_fetches_each_snapshot_artifact_once() {
        let tmp = tempfile::tempdir().unwrap();
        let commit = "fedcba9876543210fedcba9876543210fedcba98";
        let snapshot_root = tmp.path().join("snapshots").join(commit);
        std::fs::create_dir_all(&snapshot_root).unwrap();
        std::fs::write(
            snapshot_root.join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":7}"#,
        )
        .unwrap();
        std::fs::write(snapshot_root.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(
            snapshot_root.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors","c":"model-00001.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(snapshot_root.join("model-00001.safetensors"), b"").unwrap();
        std::fs::write(snapshot_root.join("model-00002.safetensors"), b"").unwrap();
        std::fs::write(
            snapshot_root.join("generation_config.json"),
            br#"{"eos_token_id":[7,8]}"#,
        )
        .unwrap();
        let files = BTreeSet::from([
            "config.json".to_string(),
            "tokenizer.json".to_string(),
            "model.safetensors.index.json".to_string(),
            "model-00001.safetensors".to_string(),
            "model-00002.safetensors".to_string(),
            "generation_config.json".to_string(),
        ]);
        let mut hub = FakeHub {
            snapshot_root: snapshot_root.clone(),
            commit: commit.to_string(),
            files,
            resolve_calls: Vec::new(),
            get_calls: Vec::new(),
        };

        let resolved = ResolvedModel::resolve_with_hub(
            Source::Hub {
                repo: "org/sharded".to_string(),
                revision: None,
            },
            None,
            &mut hub,
        )
        .unwrap();

        assert_eq!(
            resolved.weight_paths(),
            &[
                snapshot_root.join("model-00001.safetensors"),
                snapshot_root.join("model-00002.safetensors"),
            ]
        );
        assert_eq!(
            resolved.generation_config_json(),
            Some(br#"{"eos_token_id":[7,8]}"#.as_slice())
        );
        let fetched: Vec<&str> = hub
            .get_calls
            .iter()
            .map(|(_, _, filename)| filename.as_str())
            .collect();
        assert_eq!(
            fetched,
            vec![
                "config.json",
                "tokenizer.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-00001.safetensors",
                "model-00002.safetensors",
            ]
        );
    }

    #[test]
    fn cached_remote_ref_is_pinned_to_its_snapshot_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = hf_hub::Cache::new(tmp.path().join("hub"));
        let commit = "1111111111111111111111111111111111111111";
        let cache_repo = cache.repo(hf_hub::Repo::with_revision(
            "org/offline".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        cache_repo.create_ref(commit).unwrap();
        let snapshot_root = cache_repo.pointer_path(commit);
        std::fs::create_dir_all(&snapshot_root).unwrap();
        std::fs::write(
            snapshot_root.join("config.json"),
            br#"{"torch_dtype":"float16","eos_token_id":7}"#,
        )
        .unwrap();
        std::fs::write(snapshot_root.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(snapshot_root.join("model.safetensors"), b"").unwrap();

        let resolved = ResolvedModel::resolve_cached_hub(
            "org/offline".to_string(),
            "main".to_string(),
            None,
            cache,
        )
        .unwrap();

        assert_eq!(
            resolved.identity(),
            &ModelIdentity::Hub {
                repo: "org/offline".to_string(),
                commit: commit.to_string(),
            }
        );
        assert_eq!(resolved.config_path(), snapshot_root.join("config.json"));
        assert_eq!(
            resolved.tokenizer_path(),
            snapshot_root.join("tokenizer.json")
        );
        assert_eq!(resolved.dtype(), DType::F16);
    }

    #[test]
    fn unsupported_requested_dtype_fails_instead_of_falling_back() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":7}"#,
        )
        .unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), b"").unwrap();

        let error =
            ResolvedModel::resolve(Source::Local(tmp.path().to_path_buf()), Some(DType::F64))
                .unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("requested dtype F64"), "got: {message}");
        assert!(message.contains("BF16, F16, F32"), "got: {message}");
    }

    #[test]
    fn special_token_ids_come_from_resolved_metadata_and_tokenizer() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":2,"bos_token_id":1,"pad_token_id":0}"#,
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("generation_config.json"),
            br#"{"eos_token_id":[2,3]}"#,
        )
        .unwrap();
        write_tokenizer(&tmp.path().join("tokenizer.json"));
        std::fs::write(tmp.path().join("model.safetensors"), b"").unwrap();
        let resolved =
            ResolvedModel::resolve(Source::Local(tmp.path().to_path_buf()), None).unwrap();

        let tokenizer = resolved.load_tokenizer().unwrap();
        let special = resolved.special_token_ids(&tokenizer).unwrap();

        assert_eq!(special.eos(), &[2, 3]);
        assert_eq!(special.bos(), Some(1));
        assert_eq!(special.pad(), Some(0));
        assert_eq!(
            tokenizer.id_to_token(special.eos()[0]).as_deref(),
            Some("<eos>")
        );
        assert_eq!(
            tokenizer.id_to_token(special.eos()[1]).as_deref(),
            Some("<stop>")
        );
    }

    #[test]
    fn remote_missing_artifact_reports_identity_and_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let commit = "2222222222222222222222222222222222222222";
        let mut hub = FakeHub {
            snapshot_root: tmp.path().to_path_buf(),
            commit: commit.to_string(),
            files: BTreeSet::from(["config.json".to_string(), "model.safetensors".to_string()]),
            resolve_calls: Vec::new(),
            get_calls: Vec::new(),
        };

        let error = ResolvedModel::resolve_with_hub(
            Source::Hub {
                repo: "org/incomplete".to_string(),
                revision: Some("release".to_string()),
            },
            None,
            &mut hub,
        )
        .unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("org/incomplete"), "got: {message}");
        assert!(message.contains(commit), "got: {message}");
        assert!(message.contains("tokenizer.json"), "got: {message}");
    }

    #[test]
    fn special_token_missing_from_tokenizer_reports_model_identity() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":99}"#,
        )
        .unwrap();
        write_tokenizer(&tmp.path().join("tokenizer.json"));
        std::fs::write(tmp.path().join("model.safetensors"), b"").unwrap();
        let resolved =
            ResolvedModel::resolve(Source::Local(tmp.path().to_path_buf()), None).unwrap();
        let tokenizer = resolved.load_tokenizer().unwrap();

        let error = resolved.special_token_ids(&tokenizer).unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("local:"), "got: {message}");
        assert!(message.contains("eos_token_id=99"), "got: {message}");
        assert!(message.contains("tokenizer.json"), "got: {message}");
    }

    #[test]
    fn local_index_cannot_escape_the_resolved_identity_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("model");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            br#"{"torch_dtype":"bfloat16","eos_token_id":2}"#,
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(tmp.path().join("outside.safetensors"), b"").unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            br#"{"weight_map":{"w":"../outside.safetensors"}}"#,
        )
        .unwrap();

        let error = ResolvedModel::resolve(Source::Local(root), None).unwrap_err();
        let message = format!("{error:#}");

        assert!(
            message.contains("outside resolved model identity"),
            "got: {message}"
        );
        assert!(message.contains("../outside.safetensors"), "got: {message}");
    }

    #[test]
    fn unsupported_remote_dtype_fails_before_any_artifact_download() {
        let tmp = tempfile::tempdir().unwrap();
        let mut hub = FakeHub {
            snapshot_root: tmp.path().to_path_buf(),
            commit: "3333333333333333333333333333333333333333".to_string(),
            files: BTreeSet::from([
                "config.json".to_string(),
                "tokenizer.json".to_string(),
                "model.safetensors".to_string(),
            ]),
            resolve_calls: Vec::new(),
            get_calls: Vec::new(),
        };

        let error = ResolvedModel::resolve_with_hub(
            Source::Hub {
                repo: "org/model".to_string(),
                revision: None,
            },
            Some(DType::F64),
            &mut hub,
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("requested dtype F64"));
        assert!(
            hub.get_calls.is_empty(),
            "unsupported dtype downloaded artifacts"
        );
    }
}
