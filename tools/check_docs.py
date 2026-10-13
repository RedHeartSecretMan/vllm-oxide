#!/usr/bin/env python3
"""Check local links in Git-tracked Markdown, without network or dependencies.

Supports the repository's inline/reference links, ATX/setext headings and HTML
anchors. Fenced code and inline code are ignored; this is not a Markdown renderer.
"""

import argparse
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit


def without_fences(text: str) -> str:
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and marker:
            fence = marker[1]
            lines.append("\n")
        elif fence is not None:
            if re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = None
            lines.append("\n")
        else:
            lines.append(line)
    return "".join(lines)


def heading_anchors(text: str) -> set[str]:
    text = without_fences(text)
    anchors, occurrences = set(), {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        heading = match[1] if match else None
        if (
            heading is None
            and index + 1 < len(lines)
            and line.strip()
            and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[index + 1])
        ):
            heading = line.strip()
        if heading is None:
            continue
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading).replace("`", "").lower()
        slug = "".join(
            char for char in heading if char in "-_ " or unicodedata.category(char)[0] not in "PS"
        )
        slug = slug.replace(" ", "-")
        count = occurrences.get(slug, 0)
        occurrences[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count}"
        while anchor in anchors:
            count += 1
            occurrences[slug] = count + 1
            anchor = f"{slug}-{count}"
        anchors.add(anchor)
    anchors.update(re.findall(r'<(?:a|h[1-6])\b[^>]*(?:id|name)=[\'"]([^\'"]+)', text, re.I))
    return anchors


def local_links(text: str):
    text = without_fences(text)
    text = re.sub(r"(`+).*?\1", lambda match: "\n" * match[0].count("\n"), text, flags=re.S)
    references = {}
    for match in re.finditer(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)", text, re.M):
        references[" ".join(match[1].lower().split())] = match[2].strip("<>")
    pattern = (
        r"!?\[([^\]\n]*)\](?:\(\s*(<[^>]+>|(?:[^\s()]|\([^()\s]*\))+)"
        r"""(?:\s+['"][^\n]*?['"])?\s*\)|\[([^\]\n]*)\])?"""
    )
    for match in re.finditer(pattern, text):
        line = text.count("\n", 0, match.start()) + 1
        if re.match(r"\s*:", text[match.end() :]):
            continue
        if match[2] is not None:
            yield line, match[2].strip("<>")
        else:
            label = " ".join((match[3] or match[1]).lower().split())
            if label in references:
                yield line, references[label]
            elif match[3] is not None:
                yield line, None


def check_documents(root: Path, documents: list[Path]) -> list[str]:
    root = root.resolve()
    errors, anchors = [], {}
    if not documents:
        return ["no Git-tracked Markdown documents found"]
    for document in documents:
        for line, link in local_links(document.read_text(encoding="utf-8")):
            prefix = f"{document.relative_to(root)}:{line}"
            if link is None:
                errors.append(f"{prefix}: undefined link reference")
                continue
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            path = unquote(parsed.path)
            target = document
            if path:
                target = root / path.lstrip("/") if path.startswith("/") else document.parent / path
            target = target.resolve()
            if not target.is_relative_to(root):
                errors.append(f"{prefix}: local link leaves repository: {link}")
            elif not target.exists():
                errors.append(f"{prefix}: missing target: {link}")
            elif parsed.fragment and target.is_file() and target.suffix.lower() == ".md":
                if target not in anchors:
                    anchors[target] = heading_anchors(target.read_text(encoding="utf-8"))
                if unquote(parsed.fragment) not in anchors[target]:
                    errors.append(f"{prefix}: missing heading anchor: {link}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        check=True,
        capture_output=True,
    ).stdout
    documents = [root / name.decode("utf-8") for name in tracked.split(b"\0") if name]
    errors = check_documents(root, documents)
    for error in errors:
        print(error)
    print(f"Checked {len(documents)} tracked Markdown documents; {len(errors)} errors.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
