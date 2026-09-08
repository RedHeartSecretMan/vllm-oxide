import subprocess

import pytest

from golden_gen.layered_publication import publish


def test_missing_independent_permission_never_calls_transport(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_OXIDE_ALLOW_GOLDEN_PUBLISH", raising=False)

    class ForbiddenTransport:
        def ensure_absent(self):
            pytest.fail("transport called before independent permission")

    with pytest.raises(ValueError, match="independent"):
        publish(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, "a" * 40, ForbiddenTransport())


def test_tag_transport_uploads_local_candidate_objects_without_a_remote_branch(
    tmp_path, monkeypatch
):
    from golden_gen import layered_publication as module

    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    repo.mkdir()

    def git(path, *args, check=True):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=check
        )

    git(repo, "init", "-q")
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    (repo / "report").write_text("synthetic local final report\n")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=CPU",
        "-c",
        "user.email=cpu@example.invalid",
        "commit",
        "-qm",
        "local candidate",
    )
    candidate = git(repo, "rev-parse", "HEAD").stdout.strip()
    assert git(remote, "cat-file", "-e", candidate, check=False).returncode != 0
    monkeypatch.setattr(module, "PUSH_URL", str(remote), raising=False)
    monkeypatch.setattr(module.GitHubTransport, "_gh", staticmethod(lambda *args: b"{}"))
    transport = module.GitHubTransport()
    transport.repo = repo
    transport.create_tag(candidate)
    assert git(remote, "cat-file", "-e", candidate, check=False).returncode == 0
    assert git(remote, "for-each-ref", "--format=%(refname)").stdout.splitlines() == [
        "refs/tags/goldens-v0.2"
    ]
    (repo / "report").write_text("another candidate\n")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=CPU",
        "-c",
        "user.email=cpu@example.invalid",
        "commit",
        "-qm",
        "different local candidate",
    )
    another = git(repo, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(subprocess.CalledProcessError):
        transport.create_tag(another)
    assert git(remote, "rev-parse", "refs/tags/goldens-v0.2").stdout.strip() == candidate
