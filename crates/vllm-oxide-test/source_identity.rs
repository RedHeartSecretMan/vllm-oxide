//! Shared build/runtime identity of executable inputs (Git blobs, not supplied labels).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn evidence_path(path: &str) -> bool {
    matches!(
        path,
        "CONTEXT.md"
            | ".dag/definition-index.json"
            | ".dag/definitions/v0.2.0-github.json"
            | "docs/releases/goldens-v0.2-calibration-observation.json"
            | "docs/releases/goldens-v0.2.md"
    ) || (path.starts_with("docs/adr/") && path.ends_with(".md"))
}

pub fn source_files(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--cached"])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(
            "cannot enumerate tracked build inputs",
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(std::io::Error::other)?;
    Ok(text
        .split('\0')
        .filter(|path| !path.is_empty() && !evidence_path(path))
        .map(PathBuf::from)
        .collect())
}

pub fn fingerprint(root: &Path) -> Result<String, std::io::Error> {
    let mut input = Vec::new();
    for path in source_files(root)? {
        input.extend_from_slice(path.as_os_str().as_encoded_bytes());
        input.push(0);
        let contents = std::fs::read(root.join(&path))?;
        input.extend_from_slice(&(contents.len() as u64).to_le_bytes());
        input.extend_from_slice(&contents);
    }
    let mut child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("missing Git stdin"))?
        .write_all(&input)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("cannot hash build inputs"));
    }
    Ok(String::from_utf8(output.stdout)
        .map_err(std::io::Error::other)?
        .trim()
        .to_owned())
}
