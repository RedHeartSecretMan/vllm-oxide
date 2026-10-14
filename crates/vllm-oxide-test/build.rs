mod source_identity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crate_root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = crate_root.join("../..").canonicalize()?;
    // Release source archives have no Git metadata. They can still build the
    // CPU tools, but cannot supply the build receipt required by GPU evidence.
    let git_metadata = root.join(".git");
    println!("cargo:rerun-if-changed={}", git_metadata.display());
    if !git_metadata.exists() {
        println!("cargo:rustc-env=VLLM_OXIDE_BUILD_SOURCE_ID=unavailable-without-git");
        return Ok(());
    }
    for file in source_identity::source_files(&root)? {
        println!("cargo:rerun-if-changed={}", root.join(file).display());
    }
    // A newly tracked input must also invalidate the build receipt.
    let index = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["rev-parse", "--path-format=absolute", "--git-path", "index"])
        .output()?;
    println!(
        "cargo:rerun-if-changed={}",
        String::from_utf8(index.stdout)?.trim()
    );
    println!(
        "cargo:rustc-env=VLLM_OXIDE_BUILD_SOURCE_ID={}",
        source_identity::fingerprint(&root)?
    );
    Ok(())
}
