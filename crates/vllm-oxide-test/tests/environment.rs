//! The direct authoritative CLI must reject an invalid process before model work.

use std::process::Command;

#[test]
fn authoritative_cli_rejects_missing_or_wrong_deterministic_environment() -> anyhow::Result<()> {
    for (hash_seed, workspace) in [
        (None, Some(":4096:8")),
        (Some("0"), None),
        (Some("1"), Some(":4096:8")),
        (Some("0"), Some(":16:8")),
        (Some("0"), Some(":4096:8")),
    ] {
        let directory = tempfile::tempdir()?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_vllm-oxide-test"));
        command.args([
            "--mode",
            "authoritative",
            "--measurement-commit",
            "0000000000000000000000000000000000000000",
            "--measurement-tree",
            "0000000000000000000000000000000000000000",
            "--approved-observation",
            "missing-observation.json",
            "--manifest",
            "missing-manifest.json",
        ]);
        command
            .arg("--repo-root")
            .arg(directory.path())
            .arg("--model-path")
            .arg(directory.path().join("missing-model"))
            .arg("--capture-dir")
            .arg(directory.path().join("capture"))
            .env_remove("PYTHONHASHSEED")
            .env_remove("CUBLAS_WORKSPACE_CONFIG");
        if let Some(value) = hash_seed {
            command.env("PYTHONHASHSEED", value);
        }
        if let Some(value) = workspace {
            command.env("CUBLAS_WORKSPACE_CONFIG", value);
        }
        let output = command.output()?;
        assert!(!output.status.success());
        assert_eq!(
            String::from_utf8(output.stderr)?.contains("deterministic process environment"),
            hash_seed != Some("0") || workspace != Some(":4096:8")
        );
        assert!(!directory.path().join("capture").exists());
    }
    Ok(())
}

#[test]
fn benchmark_cli_rejects_missing_deterministic_environment() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let output = Command::new(env!("CARGO_BIN_EXE_vllm-oxide-benchmark"))
        .args([
            "--measurement-commit",
            "0000000000000000000000000000000000000000",
            "--measurement-tree",
            "0000000000000000000000000000000000000000",
            "--prompts-dir",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tools/golden-gen/prompts"
            ),
        ])
        .arg("--repo-root")
        .arg(directory.path())
        .arg("--model-path")
        .arg(directory.path().join("missing-model"))
        .arg("--output")
        .arg(directory.path().join("benchmark.json"))
        .env_remove("PYTHONHASHSEED")
        .env_remove("CUBLAS_WORKSPACE_CONFIG")
        .output()?;
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("deterministic process environment"));
    assert!(!directory.path().join("benchmark.json").exists());
    Ok(())
}
