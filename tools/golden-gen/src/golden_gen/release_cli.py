"""Source-bound schema5 local stages and separately authorized publication."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from golden_gen.layered_artifacts import atomic_json, sha, source_identity
from golden_gen.layered_publication import GitHubTransport, publish
from golden_gen.release_adapter import prepare_bundle, render_report, verify_bundle
from golden_gen.release_cpu import collect_cpu
from golden_gen.release_performance import collect_performance
from golden_gen.release_roles import FreezeBinding
from golden_gen.release_transport import Entrypoints


def _write_report(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _marker(
    path: Path, repo: Path, stage: str, artifacts: list[Path], result: dict[str, Any]
) -> None:
    atomic_json(
        path,
        dict(
            protocol="layered-accuracy-v1",
            schema_version=5,
            stage=stage,
            source=source_identity(repo),
            artifacts=[dict(path=str(p.resolve()), sha256=sha(p)) for p in artifacts],
            result=result,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("cpu", "performance", "bundle", "verify", "report", "publish")
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    for name in (
        "run-dir",
        "output",
        "python",
        "worker-python",
        "rust-binary",
        "target-dir",
        "model-dir",
        "benchmark-binary",
        "measurement-repo",
        "authoritative-manifest",
        "authoritative-marker",
        "performance-evidence",
        "cpu-evidence",
        "bundle-dir",
        "cache-dir",
        "review",
        "source-role-freeze",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--review-base")
    parser.add_argument("--source-role-freeze-sha256")
    args = parser.parse_args()

    def required(*names: str) -> None:
        if any(getattr(args, name) is None for name in names):
            parser.error("missing required stage arguments: " + ", ".join(names))

    try:
        if (args.source_role_freeze is None) != (args.source_role_freeze_sha256 is None):
            raise ValueError("source-role freeze requires its independently approved expected hash")
        freeze = (
            FreezeBinding(args.source_role_freeze, args.source_role_freeze_sha256)
            if args.source_role_freeze is not None
            else None
        )
        if freeze is not None and args.action not in ("performance", "bundle", "verify", "report"):
            raise ValueError("source-role routing is limited to the local release stages")
        if args.marker.exists() or args.marker.is_symlink():
            raise FileExistsError("stage completion marker must be fresh")
        if args.marker.resolve().is_relative_to(args.repo_root.resolve()):
            raise ValueError("stage markers must remain outside the reviewed source")
        result: dict[str, Any]
        files: list[Path] = []
        if args.action == "cpu":
            required("output", "python", "worker_python", "rust_binary", "target_dir")
            path = collect_cpu(
                args.repo_root,
                args.output,
                args.python,
                args.worker_python,
                args.rust_binary,
                args.target_dir,
            )
            result = dict(accepting=False, recorded_cpu_evidence=str(path))
            files = [path]
        elif args.action == "performance":
            required("output", "model_dir", "benchmark_binary", "authoritative_manifest")
            performance_entries = None
            if freeze is not None:
                required("run_dir", "authoritative_marker", "cpu_evidence", "measurement_repo")
                performance_entries = Entrypoints(
                    authoritative_manifest=args.authoritative_manifest.relative_to(
                        args.run_dir
                    ).as_posix(),
                    authoritative_marker=args.authoritative_marker.relative_to(
                        args.run_dir
                    ).as_posix(),
                    performance=(args.output / "evidence.json")
                    .relative_to(args.run_dir)
                    .as_posix(),
                    cpu_gates=args.cpu_evidence.relative_to(args.run_dir).as_posix(),
                )
            path = collect_performance(
                args.repo_root,
                args.output,
                args.model_dir,
                args.benchmark_binary,
                args.authoritative_manifest,
                measurement_repo=args.measurement_repo,
                freeze=freeze,
                evidence_root=args.run_dir,
                entrypoints=performance_entries,
            )
            result = dict(accepting=False, recorded_performance=str(path))
            files = [path]
        elif args.action == "bundle":
            required(
                "run_dir",
                "output",
                "authoritative_manifest",
                "authoritative_marker",
                "performance_evidence",
                "cpu_evidence",
            )
            entries = {
                key: getattr(args, value).relative_to(args.run_dir).as_posix()
                for key, value in {
                    "authoritative_manifest": "authoritative_manifest",
                    "authoritative_marker": "authoritative_marker",
                    "performance": "performance_evidence",
                    "cpu_gates": "cpu_evidence",
                }.items()
            }
            path = prepare_bundle(args.repo_root, args.run_dir, args.output, entries, freeze=freeze)
            result = dict(accepting=False, bundle=str(path))
            files = [path / "manifest.json", path / "goldens-v0.2.tar.gz"]
        elif args.action in ("verify", "report"):
            required("bundle_dir", "cache_dir", "rust_binary")
            verified = verify_bundle(
                args.repo_root, args.bundle_dir, args.cache_dir, args.rust_binary, freeze=freeze
            )
            files = [args.bundle_dir / "manifest.json", args.bundle_dir / "goldens-v0.2.tar.gz"]
            if args.action == "report":
                required("output")
                _write_report(args.output, render_report(verified))
                files.append(args.output)
            result = {
                key: verified[key]
                for key in (
                    "protocol",
                    "schema_version",
                    "accepting",
                    "verdict",
                    "manifest_sha256",
                    "archive_sha256",
                )
            }
        else:
            required("bundle_dir", "cache_dir", "rust_binary", "review", "review_base")
            result = publish(
                args.repo_root,
                args.bundle_dir,
                args.cache_dir,
                args.rust_binary,
                args.review,
                args.review_base,
                GitHubTransport(args.repo_root),
            )
            files = [
                args.bundle_dir / "manifest.json",
                args.bundle_dir / "goldens-v0.2.tar.gz",
                args.review,
            ]
        _marker(args.marker, args.repo_root, args.action, files, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                dict(
                    protocol="layered-accuracy-v1",
                    schema_version=5,
                    accepting=False,
                    verdict="INVALID",
                    reasons=[str(error)],
                )
            )
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
