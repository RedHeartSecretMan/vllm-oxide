import json
from pathlib import Path

import pytest

from golden_gen.prompts import discover_fixtures, load_canonical, load_prompts, load_regression
from golden_gen.schema import PromptSpec

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class TestLoadPrompts:
    def test_load_all(self):
        prompts = load_prompts(PROMPTS_DIR)
        assert len(prompts) == 25  # 5 canonical + 20 regression

    def test_all_ids_unique(self):
        prompts = load_prompts(PROMPTS_DIR)
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids))

    def test_canonical_count(self):
        prompts = load_canonical(PROMPTS_DIR)
        assert len(prompts) == 5

    def test_regression_count(self):
        prompts = load_regression(PROMPTS_DIR)
        assert len(prompts) == 20

    def test_canonical_ids_pattern(self):
        prompts = load_canonical(PROMPTS_DIR)
        for p in prompts:
            assert p.id.startswith("canonical_")
            assert p.category == "canonical"

    def test_regression_ids_pattern(self):
        prompts = load_regression(PROMPTS_DIR)
        for p in prompts:
            assert p.id.startswith("regression_")
            assert p.category == "regression"

    def test_all_have_content(self):
        prompts = load_prompts(PROMPTS_DIR)
        for p in prompts:
            assert len(p.prompt) > 0, f"Prompt {p.id} has empty content"
            assert len(p.description) > 0, f"Prompt {p.id} has empty description"

    def test_canonical_05_is_batch(self):
        prompts = load_canonical(PROMPTS_DIR)
        c05 = next(p for p in prompts if p.id == "canonical_05")
        assert c05.is_batch
        assert c05.sub_prompts is not None
        assert len(c05.sub_prompts) == 4

    def test_chat_template_flag(self):
        prompts = load_canonical(PROMPTS_DIR)
        c03 = next(p for p in prompts if p.id == "canonical_03")
        assert c03.chat_template is True

    def test_non_existent_dir(self):
        prompts = load_prompts(Path("/nonexistent/path"))
        assert len(prompts) == 0

    def test_canonical_03_length_heuristic(self):
        """Spec: canonical_03 is medium chat-templated ~200 tok.
        At ~3 chars/tok for English, 200 tok requires ~600+ chars.
        """
        prompts = load_canonical(PROMPTS_DIR)
        c03 = next(p for p in prompts if p.id == "canonical_03")
        assert len(c03.prompt) > 600, (
            f"canonical_03 prompt too short ({len(c03.prompt)} chars) for ~200 tok target"
        )

    def test_invalid_jsonl_line(self, tmp_path):
        bad_file = tmp_path / "canonical.jsonl"
        bad_file.write_text("{invalid json}\n")
        with pytest.raises(json.JSONDecodeError):
            load_canonical(tmp_path)

    def test_discovery_rejects_missing_required_fixture_family(self):
        prompts = [
            PromptSpec(
                id="canonical_01",
                category="canonical",
                prompt="hello",
                description="single only",
            )
        ]

        with pytest.raises(ValueError, match="missing required fixture families"):
            discover_fixtures(prompts)

    def test_discovery_rejects_duplicate_flattened_identifiers(self):
        prompts = [
            PromptSpec(
                id="batch_01a",
                category="canonical",
                prompt="collides",
                description="single",
            ),
            PromptSpec(
                id="batch_01",
                category="canonical",
                prompt="batch",
                description="batch",
                sub_prompts=["first", "second"],
            ),
            PromptSpec(
                id="regression_01",
                category="regression",
                prompt="regression",
                description="regression",
            ),
        ]

        with pytest.raises(ValueError, match="duplicate fixture identifier: batch_01a"):
            discover_fixtures(prompts)
