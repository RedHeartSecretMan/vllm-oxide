# Version documentation maintenance

The published source archives are being refreshed with version-specific user,
developer, research and validation-tool documentation at the user's request.
This record describes the documentation operation, not new inference validation.

| Version | Original implementation | Maintained source branch |
|---|---|---|
| `v0.1.0` | `52037493b38c2d01c4c07a98d2d8f22ca68f2509` | `docs/align-v0.1.0` |
| `v0.2.0.rc.1` | `21cd848fb1256dd908a29d4f3b08596fe1dd39fc` | `docs/align-v0.2.0-rc.1` |
| `v0.2.0` | `289cb65b475bb472b5c31db2615c3a5d4cbba0db` | `docs/align-v0.2.0` |

The original annotated tag objects remain under
`archive/before-doc-refresh-2026-10-13/<version>`. Their commits, original
measurement identities, evidence assets and verdicts remain available. The
existing version tag points to a reviewed documentation descendant so GitHub's
generated source archives include the improved guides.

Each version keeps its own implementation, dependency locks and tool interfaces.
The shared standard concerns completeness and readability: runnable commands,
required environment, accepted flags, inputs, outputs, checks, limitations and
links. A command that exists only in v0.2.0 is not advertised in v0.1.0 or RC1.

See the [release guide](releases/README.md#authorized-documentation-refresh-of-an-existing-version)
for the guarded publication process and the [version history](releases/README.md#version-history)
for the original validation state. New code behavior uses a new release version;
this maintenance operation changes documentation only.

Original tag archive: [archive/before-doc-refresh-2026-10-13/v0.1.0](https://github.com/RedHeartSecretMan/vllm-oxide/tree/archive/before-doc-refresh-2026-10-13/v0.1.0).

Original tag archive: [archive/before-doc-refresh-2026-10-13/v0.2.0.rc.1](https://github.com/RedHeartSecretMan/vllm-oxide/tree/archive/before-doc-refresh-2026-10-13/v0.2.0.rc.1).

Original tag archive: [archive/before-doc-refresh-2026-10-13/v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/tree/archive/before-doc-refresh-2026-10-13/v0.2.0).
