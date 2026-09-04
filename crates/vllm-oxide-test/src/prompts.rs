use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::types::{FixtureFamily, PromptCategory};

/// A single prompt entry loaded from a JSONL prompts file.
#[derive(Debug, Clone)]
pub struct PromptEntry {
    pub id: String,
    pub family: FixtureFamily,
    pub prompt: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusPrompt {
    id: String,
    category: PromptCategory,
    prompt: String,
    #[serde(default, rename = "description")]
    _description: Option<String>,
    #[serde(default, rename = "chat_template")]
    _chat_template: bool,
    #[serde(default, rename = "note")]
    _note: Option<String>,
    #[serde(default)]
    sub_prompts: Option<Vec<String>>,
}

/// Load all canonical prompts from a JSONL file, keyed by prompt ID.
pub fn load_canonical_prompts(prompts_dir: &Path) -> Result<HashMap<String, PromptEntry>> {
    let mut prompts = load_all_prompts(prompts_dir)?;
    prompts.retain(|_, prompt| prompt.family != FixtureFamily::Regression);
    Ok(prompts)
}

/// Load and flatten every required prompt corpus into concrete fixture identifiers.
pub fn load_all_prompts(prompts_dir: &Path) -> Result<HashMap<String, PromptEntry>> {
    let canonical_path = prompts_dir.join("canonical.jsonl");
    let regression_path = prompts_dir.join("regression.jsonl");
    for path in [&canonical_path, &regression_path] {
        if !path.exists() {
            anyhow::bail!(
                "required prompt corpus not found at {}. Set --prompts-dir to the \
                 tools/golden-gen/prompts/ directory.",
                path.display()
            );
        }
    }

    let mut prompts = HashMap::new();
    for raw in load_jsonl(&canonical_path)?
        .into_iter()
        .chain(load_jsonl(&regression_path)?)
    {
        if let Some(sub_prompts) = raw.sub_prompts {
            if raw.category != PromptCategory::Canonical || sub_prompts.len() < 2 {
                anyhow::bail!("unsupported batch prompt shape for {}", raw.id);
            }
            for (index, prompt) in sub_prompts.into_iter().enumerate() {
                let index = u8::try_from(index)
                    .ok()
                    .filter(|index| *index < 26)
                    .ok_or_else(|| anyhow::anyhow!("batch {} has more than 26 prompts", raw.id))?;
                let suffix = char::from(b'a' + index);
                insert_prompt(
                    &mut prompts,
                    PromptEntry {
                        id: format!("{}{suffix}", raw.id),
                        family: FixtureFamily::Batch,
                        prompt,
                    },
                )?;
            }
        } else {
            let family = match raw.category {
                PromptCategory::Canonical => FixtureFamily::Canonical,
                PromptCategory::Regression => FixtureFamily::Regression,
            };
            insert_prompt(
                &mut prompts,
                PromptEntry {
                    id: raw.id,
                    family,
                    prompt: raw.prompt,
                },
            )?;
        }
    }

    let families: std::collections::HashSet<_> =
        prompts.values().map(|prompt| &prompt.family).collect();
    for required in [
        FixtureFamily::Canonical,
        FixtureFamily::Batch,
        FixtureFamily::Regression,
    ] {
        if !families.contains(&required) {
            anyhow::bail!("missing required fixture family: {required:?}");
        }
    }
    Ok(prompts)
}

fn load_jsonl(path: &Path) -> Result<Vec<CorpusPrompt>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading prompts from {}", path.display()))?;

    let mut prompts = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: CorpusPrompt = serde_json::from_str(line)
            .with_context(|| format!("parsing prompt line in {}", path.display()))?;
        prompts.push(entry);
    }
    Ok(prompts)
}

fn insert_prompt(map: &mut HashMap<String, PromptEntry>, prompt: PromptEntry) -> Result<()> {
    if map.insert(prompt.id.clone(), prompt.clone()).is_some() {
        anyhow::bail!("duplicate fixture identifier: {}", prompt.id);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::load_all_prompts;
    use crate::types::FixtureFamily;

    #[test]
    fn discovers_canonical_batch_and_regression_cases() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("canonical.jsonl"),
            concat!(
                "{\"id\":\"canonical_01\",\"category\":\"canonical\",\"prompt\":\"one\"}\n",
                "{\"id\":\"canonical_02\",\"category\":\"canonical\",\"prompt\":\"batch\",",
                "\"sub_prompts\":[\"two-a\",\"two-b\"]}\n"
            ),
        )
        .unwrap();
        std::fs::write(
            temp.path().join("regression.jsonl"),
            "{\"id\":\"regression_01\",\"category\":\"regression\",\"prompt\":\"three\"}\n",
        )
        .unwrap();

        let prompts = load_all_prompts(temp.path()).unwrap();

        assert_eq!(prompts.len(), 4);
        assert_eq!(prompts["canonical_01"].family, FixtureFamily::Canonical);
        assert_eq!(prompts["canonical_02a"].family, FixtureFamily::Batch);
        assert_eq!(prompts["canonical_02a"].prompt, "two-a");
        assert_eq!(prompts["canonical_02b"].prompt, "two-b");
        assert_eq!(prompts["regression_01"].family, FixtureFamily::Regression);
    }
}
