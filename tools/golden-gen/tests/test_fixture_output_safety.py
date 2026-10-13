import numpy as np
import pytest

from golden_gen.generate import run_all
from golden_gen.io import save_fixture
from golden_gen.oracles.base import OracleResult
from golden_gen.prompts import load_prompts
from golden_gen.schema import PromptSpec


class CountingOracle:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        count = len(prompt.sub_prompts) if prompt.is_batch else 1
        return [
            OracleResult.for_regression(
                token_ids=np.array([1], dtype=np.int64),
                top5_indices=np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
                top5_logits=np.zeros((1, 5), dtype=np.float32),
                n_prompt_tokens=1,
            )
            for _ in range(count)
        ]

    def close(self):
        pass


def prompt(identifier, **kwargs):
    return PromptSpec(
        id=identifier, category="regression", prompt="hello", description="test", **kwargs
    )


def write_small(path, token=1):
    return save_fixture(
        path,
        np.array([token], dtype=np.int64),
        None,
        np.zeros((1, 5), dtype=np.int64),
        np.zeros((1, 5), dtype=np.float32),
        1,
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "../escaped",
        "nested/file",
        r"nested\file",
        ".",
        "..",
        "C:drive",
        "ending.",
        " white",
        "中文",
        "CON",
        "lpt1",
    ],
)
def test_prompt_id_rejects_path_and_device_names(identifier):
    with pytest.raises(ValueError, match="fixture identifiers"):
        prompt(identifier)


@pytest.mark.parametrize("ids", [("case", "case"), ("Case", "case")])
def test_duplicate_names_fail_before_output_or_generation(tmp_path, ids):
    oracle = CountingOracle()
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="duplicate fixture identifier"):
        run_all([oracle], [prompt(i) for i in ids], output)
    assert oracle.calls == 0
    assert not output.exists()


def test_batch_expansion_collision_fails_before_generation(tmp_path):
    batch = PromptSpec(
        id="case", category="canonical", prompt="batch", description="test", sub_prompts=["A", "B"]
    )
    oracle = CountingOracle()
    with pytest.raises(ValueError, match="duplicate fixture identifier"):
        run_all([oracle], [batch, prompt("casea")], tmp_path)
    assert oracle.calls == 0


@pytest.mark.parametrize("count", [0, 1, 27])
def test_invalid_batch_width_is_rejected(count):
    with pytest.raises(ValueError, match="2 to 26"):
        PromptSpec(
            id="batch",
            category="canonical",
            prompt="batch",
            description="test",
            sub_prompts=["A"] * count,
        )


def test_model_construct_cannot_bypass_identifier_check(tmp_path):
    oracle = CountingOracle()
    unsafe = PromptSpec.model_construct(
        id="../escaped", category="regression", prompt="x", description="test", sub_prompts=None
    )
    with pytest.raises(ValueError, match="fixture identifiers"):
        run_all([oracle], [unsafe], tmp_path)
    assert oracle.calls == 0
    assert not (tmp_path.parent / "escaped.fake.safetensors").exists()


@pytest.mark.parametrize("name", ["../oracle", r"a\b", "unknown", "CON"])
def test_invalid_oracle_name_fails_before_generation(tmp_path, name):
    oracle = CountingOracle()
    oracle.name = name
    with pytest.raises(ValueError):
        run_all([oracle], [prompt("case")], tmp_path)
    assert oracle.calls == 0


def test_duplicate_oracle_names_fail_before_generation(tmp_path):
    oracle = CountingOracle()
    with pytest.raises(ValueError, match="duplicate oracle name"):
        run_all([oracle, oracle], [prompt("case")], tmp_path)
    assert oracle.calls == 0


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_linked_existing_target_rejected_without_touching_source(tmp_path, link_type):
    source = tmp_path / "protected"
    source.write_bytes(b"keep me")
    output = tmp_path / "output"
    output.mkdir()
    target = output / "case.fake.safetensors"
    if link_type == "symlink":
        target.symlink_to(source)
    else:
        target.hardlink_to(source)
    oracle = CountingOracle()
    with pytest.raises((ValueError, OSError)):
        run_all([oracle], [prompt("case")], output)
    assert oracle.calls == 0
    with pytest.raises((ValueError, OSError)):
        write_small(target)
    assert source.read_bytes() == b"keep me"


def test_symlinked_output_ancestor_rejected(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    link = tmp_path / "link"
    link.symlink_to(protected, target_is_directory=True)
    oracle = CountingOracle()
    with pytest.raises(OSError):
        run_all([oracle], [prompt("case")], link / "output")
    assert oracle.calls == 0
    assert list(protected.iterdir()) == []


def test_dangling_target_symlink_rejected(tmp_path):
    target = tmp_path / "case.fake.safetensors"
    outside = tmp_path.parent / f"{tmp_path.name}-escaped"
    target.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        write_small(target)
    assert not outside.exists()


def test_existing_bytes_are_immutable_and_identical_retry_is_idempotent(tmp_path):
    target = tmp_path / "case.fake.safetensors"
    original_hash = write_small(target)
    status = target.stat()
    assert write_small(target) == original_hash
    assert (target.stat().st_ino, target.stat().st_mtime_ns) == (status.st_ino, status.st_mtime_ns)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_small(target, token=2)
    assert target.stat().st_mtime_ns == status.st_mtime_ns


def test_run_all_deterministic_retry_preserves_existing_fixture(tmp_path):
    oracle = CountingOracle()
    first = run_all([oracle], [prompt("case")], tmp_path)
    target = tmp_path / first[0].filename
    status = target.stat()
    second = run_all([oracle], [prompt("case")], tmp_path)
    assert first == second
    assert target.stat().st_mtime_ns == status.st_mtime_ns


def test_direct_write_rejects_parent_traversal(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        write_small(child / ".." / "escaped.safetensors")
    assert not (tmp_path / "escaped.safetensors").exists()


def test_corpus_loading_rejects_cross_file_case_collision(tmp_path):
    a = prompt("Case")
    b = prompt("case")
    (tmp_path / "canonical.jsonl").write_text(a.model_dump_json() + "\n")
    (tmp_path / "regression.jsonl").write_text(b.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="duplicate fixture identifier"):
        load_prompts(tmp_path)


def test_fifo_target_is_rejected_without_blocking_or_opening_a_writer(tmp_path):
    import os

    target = tmp_path / "case.fake.safetensors"
    os.mkfifo(target)
    with pytest.raises(ValueError, match="regular non-linked"):
        write_small(target)


def test_platform_without_no_follow_support_fails_closed(tmp_path, monkeypatch):
    import os

    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(RuntimeError, match="no-follow"):
        write_small(tmp_path / "case.fake.safetensors")
    assert list(tmp_path.iterdir()) == []


def test_fixture_bytes_match_existing_safetensors_format(tmp_path):
    from safetensors.numpy import save_file

    tensors = {
        "token_ids": np.array([1], dtype=np.int64),
        "n_prompt_tokens": np.array(1, dtype=np.int64),
        "top5_indices": np.zeros((1, 5), dtype=np.int64),
        "top5_logits": np.zeros((1, 5), dtype=np.float32),
    }
    legacy = tmp_path / "legacy.safetensors"
    save_file(tensors, str(legacy))
    hardened = tmp_path / "hardened.safetensors"
    write_small(hardened)
    assert legacy.read_bytes() == hardened.read_bytes()


def test_cleanup_does_not_hide_write_error_if_target_disappears(tmp_path, monkeypatch):
    import os

    target = tmp_path / "case.fake.safetensors"
    real_fdopen = os.fdopen

    class FailingOutput:
        def __init__(self, descriptor, mode):
            self.file = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def write(self, contents):
            temporary = next(tmp_path.glob(".case.fake.safetensors.*.tmp"))
            temporary.unlink()
            raise OSError("original write error")

        def __exit__(self, *args):
            self.file.close()

    monkeypatch.setattr(os, "fdopen", FailingOutput)
    with pytest.raises(OSError, match="original write error"):
        write_small(target)
    assert not target.exists()


def test_target_swapped_for_fifo_after_preflight_fails_without_blocking(tmp_path):
    import os

    class ReplacingOracle(CountingOracle):
        def generate(self, prompt):
            os.mkfifo(tmp_path / "case.fake.safetensors")
            return super().generate(prompt)

    oracle = ReplacingOracle()
    with pytest.raises(ValueError, match="regular non-linked"):
        run_all([oracle], [prompt("case")], tmp_path)
    assert oracle.calls == 1


@pytest.mark.parametrize("mode", ["wb", "rb"])
def test_fdopen_failure_closes_descriptor_and_keeps_existing_bytes(tmp_path, monkeypatch, mode):
    import errno
    import os

    target = tmp_path / "case.fake.safetensors"
    original = None
    if mode == "rb":
        write_small(target)
        original = target.read_bytes()
    descriptors = []
    real_fdopen = os.fdopen

    def fail_fdopen(descriptor, requested_mode):
        if requested_mode == mode:
            descriptors.append(descriptor)
            raise OSError("injected fdopen error")
        return real_fdopen(descriptor, requested_mode)

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="injected fdopen error"):
        write_small(target)
    assert len(descriptors) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert list(tmp_path.glob(".*.tmp")) == []
    if original is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == original


@pytest.mark.parametrize("failure_index", [1, 2])
def test_fstat_failure_closes_descriptor_and_cleans_only_its_temporary_file(
    tmp_path, monkeypatch, failure_index
):
    import errno
    import os

    target = tmp_path / "case.fake.safetensors"
    original = None
    if failure_index == 2:
        write_small(target)
        original = target.read_bytes()
    real_fstat = os.fstat
    descriptors = []
    calls = 0

    def fail_fstat(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_index:
            descriptors.append(descriptor)
            raise OSError("injected fstat error")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_fstat)
    with pytest.raises(OSError, match="injected fstat error"):
        write_small(target)
    assert len(descriptors) == 1
    with pytest.raises(OSError) as closed:
        real_fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert list(tmp_path.glob(".*.tmp")) == []
    if original is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == original


def test_process_interruption_never_exposes_partial_fixture_and_retry_succeeds(tmp_path):
    import os
    import subprocess
    import sys

    target = tmp_path / "case.fake.safetensors"
    script = """
import os
import sys
import numpy as np
from golden_gen.io import save_fixture

real_fdopen = os.fdopen
class InterruptedOutput:
    def __init__(self, descriptor, mode):
        self.file = real_fdopen(descriptor, mode)
    def __enter__(self):
        return self
    def write(self, contents):
        self.file.write(contents[:4])
        self.file.flush()
        os._exit(73)
    def __exit__(self, *args):
        self.file.close()
os.fdopen = InterruptedOutput
save_fixture(sys.argv[1], np.array([1], dtype=np.int64), None,
             np.zeros((1, 5), dtype=np.int64), np.zeros((1, 5), dtype=np.float32), 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        env=os.environ.copy(),
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 73, result.stderr.decode()
    assert not target.exists()
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert len(leftovers) == 1
    assert leftovers[0].stat().st_size == 4
    original_hash = write_small(target)
    from golden_gen.io import load_fixture

    np.testing.assert_array_equal(load_fixture(target)["token_ids"], [1])
    assert target.stat().st_nlink == 1
    assert write_small(target) == original_hash


def test_interruption_after_publication_leaves_single_link_and_retries_succeed(tmp_path):
    import os
    import subprocess
    import sys

    from golden_gen.io import load_fixture

    target = tmp_path / "case.fake.safetensors"
    script = """
import os
import sys
import numpy as np
import golden_gen.io as fixture_io

real_publish = fixture_io._publish_fixture_no_replace

def interrupt_after_publish(*args):
    real_publish(*args)
    os._exit(74)

fixture_io._publish_fixture_no_replace = interrupt_after_publish
fixture_io.save_fixture(sys.argv[1], np.array([1], dtype=np.int64), None,
                        np.zeros((1, 5), dtype=np.int64), np.zeros((1, 5), dtype=np.float32), 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        env=os.environ.copy(),
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 74, result.stderr.decode()
    assert target.stat().st_nlink == 1
    np.testing.assert_array_equal(load_fixture(target)["token_ids"], [1])
    assert list(tmp_path.glob(".*.tmp")) == []
    from golden_gen.io import require_regular_fixture_target

    require_regular_fixture_target(target)
    status = target.stat()
    first = write_small(target)
    assert write_small(target) == first
    assert (target.stat().st_ino, target.stat().st_mtime_ns) == (status.st_ino, status.st_mtime_ns)
    assert target.stat().st_nlink == 1


def test_atomic_publication_error_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    import errno

    import golden_gen.io as fixture_io

    target = tmp_path / "case.fake.safetensors"
    target.write_bytes(b"existing protected data")

    def unavailable(*args):
        raise OSError(errno.ENOSYS, "renameat2 unavailable")

    monkeypatch.setattr(fixture_io, "_publish_fixture_no_replace", unavailable)
    with pytest.raises(OSError, match="renameat2 unavailable"):
        write_small(target)
    assert target.read_bytes() == b"existing protected data"
    assert list(tmp_path.glob(".*.tmp")) == []
