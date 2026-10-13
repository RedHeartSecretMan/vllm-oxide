"""Behavior checks for the dependency-free documentation link checker."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_docs import check_documents


class DocumentationLinksTests(unittest.TestCase):
    def check(self, documents):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name, text in documents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                paths.append(path)
            return check_documents(root, paths)

    def test_unicode_duplicate_headings_and_reference_links(self):
        self.assertEqual(
            self.check(
                {
                    "README.md": (
                        "[中文][guide]\n[guide]: docs/guide.md#使用方式-1\n"
                        "[local](#my-heading)\n## My `Heading`\n"
                    ),
                    "docs/guide.md": (
                        "## 使用方式\n## 使用方式\n[目录](../docs/)\n"
                        "[shortcut]\n[shortcut]: ../README.md#my-heading\n"
                    ),
                }
            ),
            [],
        )

    def test_missing_targets_anchors_and_undefined_references_fail(self):
        errors = self.check(
            {"README.md": "# Start\n[missing](gone.md)\n[bad](#absent)\n[label][unknown]\n"}
        )
        self.assertEqual(len(errors), 3)
        self.assertIn("README.md:2: missing target", errors[0])
        self.assertIn("README.md:3: missing heading anchor", errors[1])
        self.assertIn("README.md:4: undefined link reference", errors[2])

    def test_code_and_external_links_are_not_checked(self):
        self.assertEqual(
            self.check(
                {
                    "README.md": (
                        "`[inline](gone.md)`\n~~~mermaid\n[diagram](absent.md)\n~~~\n"
                        "```markdown\n[example](none.md)\n```\n[web](https://example.org/nope)\n"
                        "[mail](mailto:user@example.org)\n[network](//example.org/nope)\n"
                    )
                }
            ),
            [],
        )

    def test_outside_repository_and_empty_inputs_fail(self):
        self.assertIn("leaves repository", self.check({"README.md": "[outside](../README.md)"})[0])
        with tempfile.TemporaryDirectory() as temporary:
            self.assertTrue(check_documents(Path(temporary), []))

    def test_cli_only_reads_git_tracked_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "README.md").write_text("# Start\n[local](#start)\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            for name in ("evidence.md", "target/generated.md", ".worktrees/old/README.md"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[bad](missing.md)", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_docs.py")),
                    "--root",
                    str(root),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Checked 1 tracked Markdown", result.stdout)


if __name__ == "__main__":
    unittest.main()
