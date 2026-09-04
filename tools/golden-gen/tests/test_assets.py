import gzip
import hashlib
import tarfile

import pytest

import golden_gen.assets as assets
from golden_gen.assets import _rename_no_replace, build_fixture_archive, publish_release_bundle
from golden_gen.config import COMPARISON_KERNEL_SCOPE
from golden_gen.manifest import (
    build_expected_fixtures,
    build_manifest,
    read_manifest,
    write_manifest,
)
from golden_gen.schema import (
    BaselineCalibration,
    DiscoveredFixture,
    FixtureMetadata,
    TolerancePolicy,
)
from tests.support import pinned_kernel_paths, release_runtime


def _fixture(name: str, payload: bytes, family: str = "canonical") -> FixtureMetadata:
    prompt_id, oracle, _ = name.split(".", maxsplit=2)
    return FixtureMetadata(
        prompt_id=prompt_id,
        category="regression" if family == "regression" else "canonical",
        oracle=oracle,  # type: ignore[arg-type]
        num_tokens=1,
        logits_dtype="float32",
        logits_shape=(0, 0) if family == "regression" else (1, 1),
        sha256=hashlib.sha256(payload).hexdigest(),
        filename=name,
    )


def _ready_fixture_dir(root):
    fixture_dir = root / "fixtures"
    fixture_dir.mkdir()
    cases = [
        *[(f"canonical_{index:02d}", "canonical") for index in range(1, 5)],
        *[(f"canonical_05{suffix}", "batch") for suffix in "abcd"],
        *[(f"regression_{index:02d}", "regression") for index in range(1, 21)],
    ]
    payloads = {
        f"{prompt_id}.{oracle}.safetensors": f"{prompt_id}:{oracle}".encode()
        for prompt_id, _family in cases
        for oracle in ("transformers", "vllm")
    }
    family_by_id = dict(cases)
    for name, payload in payloads.items():
        (fixture_dir / name).write_bytes(payload)
    fixtures = [
        _fixture(name, payload, family_by_id[name.split(".", maxsplit=1)[0]])
        for name, payload in payloads.items()
    ]
    archive = build_fixture_archive(
        fixture_dir,
        fixtures,
        fixture_dir / "goldens-v0.2.tar.gz",
    )
    expected = build_expected_fixtures(
        [
            DiscoveredFixture(prompt_id=prompt_id, family=family, prompt="contract")
            for prompt_id, family in cases
        ]
    )
    manifest = build_manifest(
        fixtures,
        BaselineCalibration(
            candidate_atol=0.01,
            observed_max_abs_diff=0.005,
            calibration_factor=2.0,
            method="test",
        ),
        archive=archive,
        tolerance_policy=TolerancePolicy(
            version="same-prefix-v1",
            dtype="bfloat16",
            kernel=COMPARISON_KERNEL_SCOPE,
            l1_near_tie_max_abs_logit_gap=0.02,
            l2_atol=0.01,
            rationale="reviewed test policy",
            evidence=["test:assets"],
        ),
        expected_fixtures=expected,
        runtime=release_runtime(),
        kernel_paths=pinned_kernel_paths(),
    )
    manifest.calibrated_fixtures = [f"{prompt_id}.vllm" for prompt_id, _family in cases]
    write_manifest(manifest, fixture_dir / "manifest.json")
    return fixture_dir


def test_archive_is_reproducible_ustar_with_normalized_root_entries(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    payloads = {
        "canonical_b.vllm.safetensors": b"baseline",
        "canonical_a.transformers.safetensors": b"reference",
    }
    for name, payload in payloads.items():
        (fixture_dir / name).write_bytes(payload)
    fixtures = [_fixture(name, payload) for name, payload in reversed(payloads.items())]

    first = tmp_path / "first" / "goldens-v0.2.tar.gz"
    second = tmp_path / "second" / "goldens-v0.2.tar.gz"
    first.parent.mkdir()
    second.parent.mkdir()

    first_info = build_fixture_archive(fixture_dir, fixtures, first)
    second_info = build_fixture_archive(fixture_dir, fixtures, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_info == second_info
    gzip_header = first.read_bytes()[:10]
    assert gzip_header[3] & 0x08 == 0  # no source filename
    assert gzip_header[4:8] == b"\0\0\0\0"
    with gzip.open(first, "rb") as compressed, tarfile.open(fileobj=compressed, mode="r:") as tar:
        members = tar.getmembers()
    assert [member.name for member in members] == sorted(payloads)
    assert all(
        member.isfile()
        and member.mode == 0o644
        and member.uid == 0
        and member.gid == 0
        and member.uname == ""
        and member.gname == ""
        and member.mtime == 0
        for member in members
    )


def test_atomic_publish_does_not_replace_an_empty_destination_directory(tmp_path):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "manifest.json").write_bytes(b"candidate")
    destination.mkdir()

    with pytest.raises(FileExistsError):
        _rename_no_replace(staging, destination)

    assert (staging / "manifest.json").read_bytes() == b"candidate"
    assert list(destination.iterdir()) == []


def test_atomic_publish_fails_closed_when_renameat2_is_unavailable(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    monkeypatch.setattr(assets.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(OSError, match="renameat2 is required") as raised:
        _rename_no_replace(staging, destination)

    assert raised.value.errno == assets.errno.ENOSYS
    assert staging.is_dir()
    assert not destination.exists()


def test_publisher_exposes_exactly_the_two_release_assets(tmp_path):
    fixture_dir = _ready_fixture_dir(tmp_path)

    release_dir = tmp_path / "release"
    published = publish_release_bundle(fixture_dir, release_dir)

    assert published == release_dir
    assert sorted(path.name for path in release_dir.iterdir()) == [
        "goldens-v0.2.tar.gz",
        "manifest.json",
    ]
    assert (release_dir / "manifest.json").read_bytes() == (
        fixture_dir / "manifest.json"
    ).read_bytes()
    assert (release_dir / "goldens-v0.2.tar.gz").read_bytes() == (
        fixture_dir / "goldens-v0.2.tar.gz"
    ).read_bytes()


@pytest.mark.parametrize("failure", ["missing", "unexpected", "checksum", "symlink"])
def test_publisher_rejects_incomplete_or_tampered_fixture_sets(tmp_path, failure):
    fixture_dir = _ready_fixture_dir(tmp_path)
    if failure == "missing":
        (fixture_dir / "canonical_01.vllm.safetensors").unlink()
    elif failure == "unexpected":
        (fixture_dir / "undeclared.safetensors").write_bytes(b"unexpected")
    elif failure == "checksum":
        (fixture_dir / "canonical_01.transformers.safetensors").write_bytes(b"tampered")
    else:
        fixture = fixture_dir / "canonical_01.transformers.safetensors"
        fixture.unlink()
        fixture.symlink_to(fixture_dir / "canonical_01.vllm.safetensors")
    release_dir = tmp_path / "release"

    with pytest.raises(ValueError):
        publish_release_bundle(fixture_dir, release_dir)

    assert not release_dir.exists()
    assert not any(path.name.startswith(".release.staging-") for path in tmp_path.iterdir())


def test_publisher_rejects_incomplete_calibration_and_preserves_existing_destination(tmp_path):
    fixture_dir = _ready_fixture_dir(tmp_path)
    manifest = read_manifest(fixture_dir / "manifest.json")
    manifest.calibrated_fixtures = []
    write_manifest(manifest, fixture_dir / "manifest.json")

    with pytest.raises(ValueError, match="complete baseline calibration"):
        publish_release_bundle(fixture_dir, tmp_path / "release")

    manifest.calibrated_fixtures = [
        fixture.fixture_id
        for fixture in manifest.expected_fixtures
        if fixture.oracle_role == "baseline"
    ]
    write_manifest(manifest, fixture_dir / "manifest.json")
    release_dir = tmp_path / "existing"
    release_dir.mkdir()
    sentinel = release_dir / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        publish_release_bundle(fixture_dir, release_dir)
    assert sentinel.read_bytes() == b"keep"

    dangling = tmp_path / "dangling-release"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileExistsError):
        publish_release_bundle(fixture_dir, dangling)
    assert dangling.is_symlink()
