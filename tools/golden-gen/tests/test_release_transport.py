import gzip
import hashlib
import io
import json
import os
import subprocess

import pytest

from golden_gen.release_transport import Artifact, ReleaseManifest, build_bundle, install_transport


def make_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.json").write_bytes(b"raw bytes\n")
    (source / "two.json").write_bytes(b"raw bytes\n")
    artifacts = Artifact.inventory(source, [source / "one.json", source / "two.json"])
    manifest = ReleaseManifest(
        source={"commit": "a" * 40, "tree": "b" * 40},
        registry_sha256="c" * 64,
        policy_sha256="d" * 64,
        definition_index_blob="e" * 40,
        entrypoints={
            "authoritative_manifest": "one.json",
            "authoritative_marker": "two.json",
            "performance": "one.json",
            "cpu_gates": "two.json",
        },
        counts={"artifacts": 2},
        artifacts=artifacts,
    )
    return build_bundle(source, manifest, tmp_path / "bundle")


def test_schema5_preserves_distinct_logical_files_and_installs_immutably(tmp_path):
    bundle = make_bundle(tmp_path)
    installed = install_transport(bundle, tmp_path / "cache")
    assert (installed / "evidence/one.json").read_bytes() == b"raw bytes\n"
    assert (installed / "evidence/two.json").read_bytes() == b"raw bytes\n"
    assert install_transport(bundle, tmp_path / "cache") == installed
    (installed / "evidence/one.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum|size"):
        install_transport(bundle, tmp_path / "cache")


def test_concurrent_schema5_installs_share_only_a_verified_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    bundle = make_bundle(tmp_path)
    barrier = Barrier(2)

    def install():
        barrier.wait()
        return install_transport(bundle, tmp_path / "cache")

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: install(), range(2)))
    assert results[0] == results[1]
    assert (results[0] / "evidence/one.json").read_bytes() == b"raw bytes\n"
    assert list((tmp_path / "cache/goldens-v0.2").iterdir()) == [results[0]]


def test_real_rust_consumer_reads_python_schema5_bytes(tmp_path):
    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("set GOLDEN_TRANSPORT_TEST_BINARY to the CPU verify-bundle binary")
    bundle = make_bundle(tmp_path)
    completed = subprocess.run(
        [
            binary,
            "--layered-transport",
            "--bundle-dir",
            str(bundle),
            "--cache-dir",
            str(tmp_path / "rust-cache"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["accepting"] is False


@pytest.mark.parametrize("size", [2147483648, 2147483649])
def test_current_upload_boundary_requires_a_capacity_decision(size):
    from golden_gen.release_transport import check_upload_size

    with pytest.raises(ValueError, match="under 2 GiB"):
        check_upload_size(size)
    check_upload_size(2147483647)


@pytest.mark.parametrize("mutation", ["mode", "truncated", "tar-tail", "gzip-tail", "payload"])
def test_corrupt_archive_refusals_agree_with_real_rust_consumer(tmp_path, mutation):
    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("CPU transport binary required")
    bundle = make_bundle(tmp_path)
    archive = bundle / "goldens-v0.2.tar.gz"
    tar = bytearray(gzip.decompress(archive.read_bytes()))
    if mutation == "mode":
        tar[100:108] = b"0000777\0"
        tar[148:156] = b"        "
        tar[148:156] = f"{sum(tar[:512]):06o}\0 ".encode()
    elif mutation == "tar-tail":
        tar[-1] = 1
    elif mutation == "payload":
        tar[512] ^= 1
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as output:
        output.write(tar)
    payload = stream.getvalue()
    if mutation == "truncated":
        payload = payload[:-8]
    if mutation == "gzip-tail":
        payload += b"trailing"
    archive.write_bytes(payload)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["archive"]["sha256"] = hashlib.sha256(payload).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, EOFError)):
        install_transport(bundle, tmp_path / "python-cache")
    result = subprocess.run(
        [
            binary,
            "--layered-transport",
            "--bundle-dir",
            str(bundle),
            "--cache-dir",
            str(tmp_path / "rust-cache"),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    for cache in ("python-cache", "rust-cache"):
        assert not (tmp_path / cache / "goldens-v0.2" / manifest["archive"]["sha256"]).exists()


def test_artifact_reference_cannot_hide_a_noncanonical_path(tmp_path):
    from golden_gen.layered_artifacts import bound_file, sha

    path = tmp_path / "payload"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="path"):
        bound_file(tmp_path, dict(path="./payload", sha256=sha(path)))


@pytest.mark.parametrize("mutation", ["floating-version", "missing-archive-name"])
def test_wire_types_and_required_nested_fields_agree(tmp_path, mutation):
    from golden_gen.release_transport import read_manifest

    binary = os.environ.get("GOLDEN_TRANSPORT_TEST_BINARY")
    if not binary:
        pytest.skip("CPU transport binary required")
    bundle = make_bundle(tmp_path)
    path = bundle / "manifest.json"
    raw = json.loads(path.read_text())
    if mutation == "floating-version":
        raw["schema_version"] = 5.0
    else:
        del raw["archive"]["filename"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        read_manifest(path)
    actual = subprocess.run(
        [
            binary,
            "--layered-transport",
            "--bundle-dir",
            str(bundle),
            "--cache-dir",
            str(tmp_path / "rust"),
        ],
        capture_output=True,
    )
    assert actual.returncode != 0


@pytest.mark.parametrize(
    "path", ["../bad", "/bad", "a//b", "a/./b", "a/../b", ".git/config", "a\\b"]
)
def test_logical_paths_reject_aliases_before_any_file_access(path):
    with pytest.raises(ValueError):
        Artifact(logical_path=path, filename="artifact-000000.bin", sha256="a" * 64, size_bytes=1)
