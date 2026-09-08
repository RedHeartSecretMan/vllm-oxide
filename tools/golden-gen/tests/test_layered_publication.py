import pytest

from golden_gen.layered_publication import publish


def test_missing_independent_permission_never_calls_transport(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH", raising=False)

    class ForbiddenTransport:
        def ensure_absent(self):
            pytest.fail("transport called before independent permission")

    with pytest.raises(ValueError, match="independent"):
        publish(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, "a" * 40, ForbiddenTransport())
