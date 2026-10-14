use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// A single prompt entry loaded from a JSONL prompts file.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptEntry {
    pub id: String,
    pub category: String,
    pub prompt: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub chat_template: bool,
    #[serde(default)]
    pub sub_prompts: Option<Vec<String>>,
}

/// Load all canonical prompts from a JSONL file, keyed by prompt ID.
pub fn load_canonical_prompts(prompts_dir: &Path) -> Result<HashMap<String, PromptEntry>> {
    let path = prompts_dir.join("canonical.jsonl");
    if !path.exists() {
        anyhow::bail!(
            "canonical.jsonl not found at {}. \
             Set --prompts-dir to the tools/golden-gen/prompts/ directory.",
            path.display()
        );
    }
    load_jsonl(&path)
}

fn load_jsonl(path: &Path) -> Result<HashMap<String, PromptEntry>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading prompts from {}", path.display()))?;

    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: PromptEntry = serde_json::from_str(line)
            .with_context(|| format!("parsing prompt line in {}", path.display()))?;
        if entry.id.is_empty() || map.insert(entry.id.clone(), entry).is_some() {
            anyhow::bail!("empty or duplicate prompt identifier in {}", path.display());
        }
    }
    Ok(map)
}

/// Load both corpora and expand batch members into the IDs stored in schema1 fixtures.
/// Each fixture is compared by its own single-request diagnostic capture.
pub fn load_all_prompts(prompts_dir: &Path) -> Result<HashMap<String, PromptEntry>> {
    let mut result = HashMap::new();
    for (filename, category) in [
        ("canonical.jsonl", "canonical"),
        ("regression.jsonl", "regression"),
    ] {
        for mut entry in load_jsonl(&prompts_dir.join(filename))?.into_values() {
            if entry.category != category {
                anyhow::bail!("prompt {} has incorrect corpus category", entry.id);
            }
            if let Some(members) = entry.sub_prompts.take() {
                if category != "canonical" || !(2..=26).contains(&members.len()) {
                    anyhow::bail!("unsupported batch prompt shape for {}", entry.id);
                }
                for (index, prompt) in members.into_iter().enumerate() {
                    let mut member = entry.clone();
                    member.id.push(char::from(b'a' + u8::try_from(index)?));
                    member.prompt = prompt;
                    if member.prompt.is_empty()
                        || result.insert(member.id.clone(), member).is_some()
                    {
                        anyhow::bail!("empty or duplicate expanded prompt");
                    }
                }
            } else if entry.prompt.is_empty() || result.insert(entry.id.clone(), entry).is_some() {
                anyhow::bail!("empty or duplicate prompt");
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    #[test]
    fn all_corpora_and_batch_members_are_loaded_without_silent_overwrites() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("canonical.jsonl"),
            "{\"id\":\"canonical_05\",\"category\":\"canonical\",\"prompt\":\"batch\",\"sub_prompts\":[\"one\",\"two\"]}\n").unwrap();
        let regression =
            "{\"id\":\"regression_01\",\"category\":\"regression\",\"prompt\":\"three\"}\n";
        std::fs::write(root.path().join("regression.jsonl"), regression).unwrap();
        let prompts = super::load_all_prompts(root.path()).unwrap();
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts["canonical_05a"].prompt, "one");
        assert_eq!(prompts["canonical_05b"].prompt, "two");
        assert_eq!(prompts["regression_01"].prompt, "three");
        std::fs::write(root.path().join("regression.jsonl"), regression.repeat(2)).unwrap();
        assert!(super::load_all_prompts(root.path()).is_err());
    }
}
