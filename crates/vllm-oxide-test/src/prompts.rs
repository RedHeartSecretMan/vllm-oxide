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
        map.insert(entry.id.clone(), entry);
    }
    Ok(map)
}
