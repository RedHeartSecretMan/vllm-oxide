//! Fail-closed binding between GPU evidence and the reviewed Git tree.

use std::io::Read;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

/// Rehash the actual local source before every GPU owner's model initialization.
pub fn validate_release_model(model_path: &Path) -> Result<()> {
    for (filename, expected) in [
        ("config.json", crate::types::MODEL_CONFIG_SHA256),
        ("tokenizer.json", crate::types::TOKENIZER_SHA256),
        ("model.safetensors", crate::types::MODEL_WEIGHTS_SHA256),
    ] {
        let mut file = std::fs::File::open(model_path.join(filename))
            .with_context(|| format!("opening release model artifact {filename}"))?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        if format!("{:x}", digest.finalize()) != expected {
            bail!("release model artifact SHA-256 differs from ADR-0012: {filename}");
        }
    }
    for entry in std::fs::read_dir(model_path)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if (name.ends_with(".safetensors") && name != "model.safetensors")
            || name.ends_with(".safetensors.index.json")
        {
            bail!("unexpected alternative weights in release model directory: {name}");
        }
    }
    Ok(())
}

#[path = "../source_identity.rs"]
mod source_identity;

/// Check the compiler-produced source receipt before a GPU owner loads the model.
pub fn validate_running_binary(repo_root: &Path) -> Result<()> {
    let live = source_identity::fingerprint(repo_root)?;
    if live != env!("VLLM_OXIDE_BUILD_SOURCE_ID") {
        bail!(
            "running binary was built from different executable inputs; rebuild the reviewed tree"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementIdentity {
    pub commit: String,
    pub tree: String,
}

/// Prove that a GPU-owning process is executing from the exact clean reviewed tree.
pub fn validate_measurement_identity(
    repo_root: &Path,
    expected_commit: &str,
    expected_tree: &str,
) -> Result<MeasurementIdentity> {
    if !is_lowercase_hex(expected_commit, 40) || !is_lowercase_hex(expected_tree, 40) {
        bail!("measurement commit and tree must be exact lowercase 40-hex identities");
    }
    let canonical_root = repo_root
        .canonicalize()
        .with_context(|| format!("canonicalizing repository root {}", repo_root.display()))?;
    let discovered_root = git_text(&canonical_root, &["rev-parse", "--show-toplevel"])?;
    let discovered_root = Path::new(&discovered_root)
        .canonicalize()
        .context("canonicalizing Git top-level directory")?;
    if discovered_root != canonical_root {
        bail!("measurement repo root is not the Git top-level directory");
    }
    let resolved_tree = git_text(
        &canonical_root,
        &["rev-parse", &format!("{expected_commit}^{{tree}}")],
    )?;
    if resolved_tree != expected_tree {
        bail!(
            "reviewed measurement commit/tree identity is invalid: \
             expected {expected_commit}/{expected_tree}, resolved tree {resolved_tree}"
        );
    }
    let current_commit = git_text(&canonical_root, &["rev-parse", "HEAD"])?;
    let ancestor = Command::new("git")
        .arg("-C")
        .arg(&canonical_root)
        .args([
            "merge-base",
            "--is-ancestor",
            expected_commit,
            &current_commit,
        ])
        .status()
        .context("checking measurement commit ancestry")?;
    if !ancestor.success() {
        bail!("reviewed measurement commit is not an ancestor of the live HEAD");
    }
    let changed = git_bytes(
        &canonical_root,
        &[
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            &format!("{expected_commit}..{current_commit}"),
            "--",
        ],
    )?;
    let disallowed = changed
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty() && !is_allowed_evidence_path(path))
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect::<Vec<_>>();
    if !disallowed.is_empty() {
        bail!(
            "executable or unreviewed bytes changed after measurement: {}",
            disallowed.join(", ")
        );
    }
    let status = Command::new("git")
        .args([
            "-C",
            canonical_root
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("repository path is not UTF-8"))?,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ])
        .output()
        .context("checking measurement worktree status")?;
    if !status.status.success() {
        bail!(
            "Git status failed while binding measurement: {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }
    if !status.stdout.is_empty() {
        bail!("measurement worktree is not clean");
    }
    Ok(MeasurementIdentity {
        commit: expected_commit.to_owned(),
        tree: expected_tree.to_owned(),
    })
}

fn is_allowed_evidence_path(path: &[u8]) -> bool {
    std::str::from_utf8(path).is_ok_and(source_identity::evidence_path)
}

fn git_text(repo_root: &Path, args: &[&str]) -> Result<String> {
    let bytes = git_bytes(repo_root, args)?;
    let value = String::from_utf8(bytes).context("Git output is not UTF-8")?;
    Ok(value.trim().to_owned())
}

fn git_bytes(repo_root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}

fn is_lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::{git_text, validate_measurement_identity, validate_release_model};

    #[test]
    fn release_model_rejects_wrong_local_bytes_before_model_loading() {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(temporary.path().join("config.json"), b"unapproved model").unwrap();
        let error = validate_release_model(temporary.path()).unwrap_err();
        assert!(error.to_string().contains("SHA-256 differs"));
    }

    fn git(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn measurement_requires_exact_clean_head_and_tree() {
        let temporary = tempfile::tempdir().unwrap();
        let repo = temporary.path();
        git(repo, &["init", "-q"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "measurement",
            ],
        );
        let commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let tree = git_text(repo, &["rev-parse", "HEAD^{tree}"]).unwrap();

        let identity = validate_measurement_identity(repo, &commit, &tree).unwrap();
        assert_eq!(identity.commit, commit);
        assert_eq!(identity.tree, tree);

        fs::write(repo.join("untracked"), b"dirty").unwrap();
        let error = validate_measurement_identity(repo, &commit, &tree).unwrap_err();
        assert!(error.to_string().contains("not clean"));
    }

    #[test]
    fn measurement_rejects_executable_renamed_into_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let repo = temporary.path();
        git(repo, &["init", "-q"]);
        git(repo, &["config", "diff.renames", "true"]);
        git(repo, &["config", "user.name", "Test"]);
        git(repo, &["config", "user.email", "test@example.com"]);
        fs::write(repo.join("runner.rs"), b"fn main() {}\n").unwrap();
        git(repo, &["add", "."]);
        git(repo, &["commit", "-qm", "measurement"]);
        let commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let tree = git_text(repo, &["rev-parse", "HEAD^{tree}"]).unwrap();
        fs::create_dir_all(repo.join("docs/adr")).unwrap();
        fs::rename(
            repo.join("runner.rs"),
            repo.join("docs/adr/renamed-source.md"),
        )
        .unwrap();
        git(repo, &["add", "-A"]);
        git(
            repo,
            &["commit", "-qm", "hide source deletion as evidence rename"],
        );

        let error = validate_measurement_identity(repo, &commit, &tree).unwrap_err();
        assert!(error.to_string().contains("unreviewed bytes"));
    }

    #[test]
    fn measurement_allows_only_reviewed_evidence_after_the_frozen_tree() {
        let temporary = tempfile::tempdir().unwrap();
        let repo = temporary.path();
        git(repo, &["init", "-q"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-qm",
                "measurement",
            ],
        );
        let commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let tree = git_text(repo, &["rev-parse", "HEAD^{tree}"]).unwrap();
        fs::create_dir_all(repo.join("docs/releases")).unwrap();
        fs::write(repo.join("docs/releases/goldens-v0.2.md"), b"evidence\n").unwrap();
        git(repo, &["add", "docs/releases/goldens-v0.2.md"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "evidence",
            ],
        );
        validate_measurement_identity(repo, &commit, &tree).unwrap();

        fs::create_dir(repo.join("src")).unwrap();
        fs::write(repo.join("src/lib.rs"), b"pub fn changed() {}\n").unwrap();
        git(repo, &["add", "src/lib.rs"]);
        git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "executable",
            ],
        );
        let error = validate_measurement_identity(repo, &commit, &tree).unwrap_err();
        assert!(error.to_string().contains("unreviewed bytes"));
    }
}
