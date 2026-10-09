# Calibration policy binding without execution-source drift

An immutable calibration snapshot may keep an unapproved numerical-policy
binding, so its confirmation worker remains sealed. After approval, a measured
descendant must bind the approved numerical policy. Its supervision document
must also update its numerical-policy SHA pointer or the complete CPU policy
checks correctly reject the mismatch.

Calibration provenance permits this one metadata change only when both
supervision documents are regular Git blobs with the same file mode, every
other canonical JSON field is identical, and each numerical-policy pointer
matches that revision's actual numerical-policy blob SHA-256. False and zero
remain different values. Missing/malformed data, symlinks, mode changes,
resource-setting changes and execution-source changes are rejected.

This evaluator rule belongs in the separately frozen supervisor. The measured
descendant contains only approved policy/index/ADR changes and must rebuild its
binaries and complete CPU gates. Calibration keeps its original source, bytes
and verdict; ancestry, registry, closure, replay and protected-execution checks
still apply. The supervisor must bind the actual new measured source and
binaries and pass its own CPU gates. This rule grants no scope or release
approval and does not open confirmation in the current calibration snapshot.
