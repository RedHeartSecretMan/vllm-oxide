import subprocess
import sys

import pytest


def test_failed_candidate_subprocess_preserves_both_log_streams(capsys) -> None:
    from golden_gen.layered_cli import run_candidate_capture

    with pytest.raises(subprocess.CalledProcessError):
        run_candidate_capture(
            [
                sys.executable,
                "-c",
                "import sys; print('retained stdout'); "
                "print('retained stderr',file=sys.stderr); sys.exit(5)",
            ]
        )
    captured = capsys.readouterr()
    assert captured.out == "retained stdout\n"
    assert captured.err == "retained stderr\n"
