"""Revalidate complete layered evidence before bundling, reporting or publication."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from golden_gen.assets import _rename_no_replace
from golden_gen.layered_artifacts import (
    bound_file,
    manifest_artifact_closure,
    sha,
    source_identity,
    verify_marker,
)
from golden_gen.layered_inventory import frozen_owner_inventory
from golden_gen.layered_manifest import LayeredManifest, evaluate_manifest
from golden_gen.layered_release import POLICY_PATH, REGISTRY_PATH, Registry, definition_document
from golden_gen.release_cpu import validate_cpu
from golden_gen.release_performance import validate_performance
from golden_gen.release_transport import (
    Artifact,
    Entrypoints,
    ReleaseManifest,
    build_bundle,
    install_transport,
    logical_path,
    read_manifest,
)

REPORT = "docs/releases/goldens-v0.2.md"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def _source_definitions(repo: Path, source: dict[str, str]) -> dict[str, bytes]:
    commit = source["commit"]
    if git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip() != source["tree"]:
        raise ValueError("Definition source commit/tree mismatch")
    prefix = f"definitions/{commit}/"
    raw = git(repo, "show", f"{commit}:.dag/definition-index.json")
    index = json.loads(raw)
    result = {prefix + "index.json": raw}
    entries = index["inputs"]
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("duplicate Definition input")
    for entry in entries:
        relative = logical_path(entry["path"])
        if git(repo, "rev-parse", f"{commit}:{relative}").decode().strip() != entry["blob_oid"]:
            raise ValueError("selected Definition blob mismatch")
        result[prefix + "selected/" + relative] = git(repo, "show", f"{commit}:{relative}")
    return result


def _sources(path: Path) -> list[dict[str, str]]:
    manifest = LayeredManifest.model_validate_json(path.read_text())
    sources = [manifest.source]
    if manifest.calibration_manifest is not None:
        sources.extend(
            _sources(bound_file(path.parent, manifest.calibration_manifest.model_dump()))
        )
    return sources


def _definitions(repo: Path, manifest: Path) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for source in _sources(manifest):
        for name, value in _source_definitions(repo, source).items():
            if name in snapshots and snapshots[name] != value:
                raise ValueError("conflicting Definition source snapshots")
            snapshots[name] = value
    return snapshots


def _authoritative(repo: Path, path: Path) -> dict[str, Any]:
    result = evaluate_manifest(repo, path, authoritative=True)
    if result.get("verdict") != "PASS" or result.get("accepting") is not True:
        raise ValueError(f"authoritative evidence is not complete PASS: {result.get('reasons')}")
    return result


def validate_evidence(
    repo: Path, root: Path, entrypoints: Entrypoints
) -> tuple[dict[str, Any], list[Path]]:
    paths = {key: root / logical_path(value) for key, value in entrypoints.model_dump().items()}
    result = _authoritative(repo, paths["authoritative_manifest"])
    source = source_identity(repo)
    marker = verify_marker(paths["authoritative_marker"], source)
    marker_root = paths["authoritative_marker"].parent.parent
    if (
        marker["stage"] != "authoritative"
        or len(marker["outputs"]) < 2
        or bound_file(marker_root, marker["outputs"][1]).resolve()
        != paths["authoritative_manifest"].resolve()
        or json.loads(bound_file(marker_root, marker["outputs"][0]).read_text()) != result
    ):
        raise ValueError(
            "authoritative marker result does not reproduce from its exact raw manifest"
        )
    registry_data, registry_sha = definition_document(repo, REGISTRY_PATH)
    policy_data, policy_sha = definition_document(repo, POLICY_PATH)
    registry = Registry.model_validate(registry_data)
    closure = [
        paths["authoritative_marker"],
        *(bound_file(marker_root, r) for r in marker["outputs"]),
        *manifest_artifact_closure(paths["authoritative_manifest"]),
    ]
    manifest = LayeredManifest.model_validate_json(paths["authoritative_manifest"].read_text())
    candidate_receipts = [
        json.loads(
            bound_file(
                paths["authoritative_manifest"].parent, e.primary_receipt.model_dump()
            ).read_text()
        )
        for e in manifest.captures
        if e.engine == "candidate"
    ]
    build_ids = {r["build_source_id"] for r in candidate_receipts}
    if len(build_ids) != 1:
        raise ValueError("candidate binary build identities conflict")
    performance, performance_files = validate_performance(
        paths["performance"], source, result["runtime_profile"], registry, next(iter(build_ids))
    )
    wrapper = json.loads(paths["performance"].read_text())
    if wrapper.get("authoritative_manifest_sha256") != sha(paths["authoritative_manifest"]):
        raise ValueError("performance predecessor differs from authoritative evidence")
    closure.extend(performance_files)
    closure.extend(validate_cpu(paths["cpu_gates"], source))
    for name, value in _definitions(repo, paths["authoritative_manifest"]).items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise ValueError("archived Definition snapshot differs from its own measured source")
        closure.append(path)
    inventory = Artifact.inventory(root, list(dict.fromkeys(closure)))
    counts = dict(
        execution_groups=len(registry.numerical_cases),
        numerical_cases=sum(len(c.plan.members) for c in registry.numerical_cases),
        prediction_rows_per_engine_variant=sum(
            len(m.continuation) for c in registry.numerical_cases for m in c.plan.members
        ),
        setup_calls_per_engine_variant=sum(len(c.setup_calls) for c in registry.numerical_cases),
        setup_rows_per_engine_variant=sum(
            len(m.continuation)
            for c in registry.numerical_cases
            for p in c.setup_calls
            for m in p.members
        ),
        unique_owners=len(frozen_owner_inventory(registry, authoritative=True)),
        operator_profiles=len(registry.operator_profiles),
        behavior_cases=len(registry.behavior_cases),
        artifacts=len(inventory),
    )
    metadata = dict(
        source=source,
        registry_sha256=registry_sha,
        policy_sha256=policy_sha,
        definition_index_blob=git(repo, "rev-parse", "HEAD:.dag/definition-index.json")
        .decode()
        .strip(),
        entrypoints=entrypoints.model_dump(),
        counts=counts,
        artifacts=[a.model_dump() for a in inventory],
    )
    return dict(
        manifest=metadata,
        result=result,
        performance=performance,
        policy=policy_data,
        model=candidate_receipts[0]["model"],
        candidate_binaries=sorted({r["binary_sha256"] for r in candidate_receipts}),
    ), closure


def prepare_bundle(repo: Path, root: Path, destination: Path, entrypoints: dict[str, str]) -> Path:
    entries = Entrypoints.model_validate(entrypoints)
    path = root / logical_path(entries.authoritative_manifest)
    _authoritative(
        repo, path
    )  # Fail before reading sealed data or creating output when budgets are pending.
    for name, value in _definitions(repo, path).items():
        output = root / name
        if output.exists() or output.is_symlink():
            if output.is_symlink() or output.read_bytes() != value:
                raise ValueError("refusing to overwrite Definition snapshot")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(value)
    evidence, _ = validate_evidence(repo, root, entries)
    manifest = ReleaseManifest.model_validate(evidence["manifest"])
    with tempfile.TemporaryDirectory(
        prefix=".release-bundle-", dir=destination.parent
    ) as directory:
        bundle = build_bundle(root, manifest, Path(directory) / "bundle")
        # Re-read unchanged originals before exposing the semantic-stage output.
        fresh, _ = validate_evidence(repo, root, entries)
        if fresh != evidence:
            raise ValueError("evidence changed during bundling")
        _rename_no_replace(bundle, destination)
    return destination


@contextmanager
def measurement_checkout(repo: Path, source: dict[str, str]) -> Iterator[Path]:
    current = source_identity(repo)
    if current == source:
        yield repo
        return
    git(repo, "merge-base", "--is-ancestor", source["commit"], current["commit"])
    changed = git(
        repo, "diff", "--no-renames", "--name-only", "-z", source["commit"], current["commit"]
    ).split(b"\0")
    if set(changed) - {b""} != {REPORT.encode()}:
        raise ValueError("post-measurement candidate changes more than the report")
    with tempfile.TemporaryDirectory(prefix="layered-source-") as directory:
        checkout = Path(directory) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(checkout)],
            check=True,
            capture_output=True,
        )
        git(checkout, "checkout", "--quiet", "--detach", source["commit"])
        if source_identity(checkout) != source:
            raise ValueError("clean measurement checkout identity mismatch")
        yield checkout


def verify_bundle(repo: Path, bundle: Path, cache: Path, rust_binary: Path) -> dict[str, Any]:
    manifest = read_manifest(bundle / "manifest.json")
    original_hashes = {p.name: sha(p) for p in bundle.iterdir()}
    with (
        measurement_checkout(repo, manifest.source.model_dump()) as measured,
        tempfile.TemporaryDirectory(prefix="layered-consumer-") as directory,
    ):
        stage = Path(directory)
        # Rust reads the actual archive, not the Python extracted tree or a PASS summary.
        completed = subprocess.run(
            [
                str(rust_binary),
                "--layered-transport",
                "--bundle-dir",
                str(bundle),
                "--cache-dir",
                str(stage / "rust"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        transport = json.loads(completed.stdout)
        installed = Path(transport["installed"])
        if (
            transport.get("transport_verified") is not True
            or transport.get("accepting") is not False
            or not installed.resolve().is_relative_to(stage.resolve())
        ):
            raise ValueError("Rust transport did not return a confined nonaccepting install")
        python_install = install_transport(bundle, stage / "python")
        if (installed / "manifest.json").read_bytes() != (
            python_install / "manifest.json"
        ).read_bytes():
            raise ValueError("Python/Rust transport manifests differ")
        evidence, _ = validate_evidence(measured, installed / "evidence", manifest.entrypoints)
        reconstructed = ReleaseManifest.model_validate(evidence["manifest"]).model_copy(
            update={"archive": manifest.archive}
        )
        if reconstructed != manifest:
            raise ValueError("release inventory/counts/identities differ from recomputed evidence")
    if {p.name: sha(p) for p in bundle.iterdir()} != original_hashes:
        raise ValueError("bundle changed during clean verification")
    installed = install_transport(bundle, cache)
    return dict(
        protocol="layered-accuracy-v1",
        schema_version=5,
        accepting=True,
        verdict="PASS",
        installed=str(installed),
        manifest_sha256=original_hashes["manifest.json"],
        archive_sha256=original_hashes["goldens-v0.2.tar.gz"],
        **evidence,
    )


def render_report(verified: dict[str, Any]) -> str:
    if verified.get("accepting") is not True or verified.get("verdict") != "PASS":
        raise ValueError("report requires revalidated complete release evidence")
    return "\n".join(
        [
            "# goldens-v0.2 layered release evidence",
            "",
            f"Measurement commit: `{verified['manifest']['source']['commit']}`",
            f"Measurement tree: `{verified['manifest']['source']['tree']}`",
            "",
            "Protocol: layered-accuracy-v1; release schema5. "
            "Final candidate identity is recorded externally.",
            "",
            "## Exact asset hashes",
            "",
            f"- `manifest.json`: `{verified['manifest_sha256']}`",
            f"- `goldens-v0.2.tar.gz`: `{verified['archive_sha256']}`",
            "",
            "## Recomputed cases, identities, budgets and original evidence hashes",
            "",
            "```json",
            json.dumps(
                {
                    k: verified[k]
                    for k in ("manifest", "result", "policy", "model", "candidate_binaries")
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            "```",
            "",
            "## Performance raw samples and headline medians",
            "",
            "```json",
            json.dumps(verified["performance"], indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
            "## Limitations",
            "",
            "Single-GPU Qwen3-0.6B BF16 under the recorded runtime only. "
            "Performance is evidence, not a hardware SLA.",
            "Fixed histories do not cover every free-generation history. "
            "CPU checks do not establish GPU numerical accuracy.",
            "The publication stage still requires a clean evidence-only candidate, "
            "fresh full review and independent user authority.",
            "",
        ]
    )
