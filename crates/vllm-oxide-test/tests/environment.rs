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

#[test]
fn release_archive_builds_without_a_git_evidence_receipt() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let crate_root = directory.path().join("crates/vllm-oxide-test");
    std::fs::create_dir_all(&crate_root)?;
    std::fs::write(crate_root.join("build.rs"), include_str!("../build.rs"))?;
    std::fs::write(
        crate_root.join("source_identity.rs"),
        include_str!("../source_identity.rs"),
    )?;
    let executable = directory.path().join("archive-build-script");
    let compiled = Command::new("rustc")
        .arg("--edition=2021")
        .arg(crate_root.join("build.rs"))
        .arg("-o")
        .arg(&executable)
        .output()?;
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let output = Command::new(&executable)
        .env("CARGO_MANIFEST_DIR", &crate_root)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout)?
        .contains("VLLM_OXIDE_BUILD_SOURCE_ID=unavailable-without-git"));
    // Compilable archives still cannot be mistaken for an authoritative Git checkout.
    let Err(error) = vllm_oxide_test::measurement::validate_measurement_identity(
        directory.path(),
        &"0".repeat(40),
        &"0".repeat(40),
    ) else {
        anyhow::bail!("an archive without Git metadata supplied an authoritative identity")
    };
    assert!(error.to_string().contains("git rev-parse"));
    Ok(())
}
