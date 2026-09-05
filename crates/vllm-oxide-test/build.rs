mod source_identity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crate_root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = crate_root.join("../..").canonicalize()?;
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
