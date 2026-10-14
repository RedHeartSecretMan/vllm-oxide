# Contributing to vllm-oxide

Documentation, code, tests and reproducible issue reports are welcome in English or Chinese. You can contribute without a GPU.

This archive documents v0.1.0. Start new contributions from the current `main` branch and its [contribution guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/CONTRIBUTING.md); the historical archive is useful for reproducing old behavior, not a separate active feature branch.

## Choose a change

- For a small documentation correction, update the relevant guide and keep the two README languages aligned.
- For a bug, search [existing issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) and include the exact source commit, command and expected/actual behavior.
- For a feature or API change, explain the use case and proposed behavior before a large implementation.
- For a ready ticket, follow its acceptance criteria and the [issue workflow](docs/agents/issue-tracker.md).

## Validate and submit

Use the [development guide](docs/development.md) for version-correct checks. Explain what changed and why, which checks actually ran, and any compatibility impact. CPU-only changes need no model download or GPU. Inference changes require relevant GPU validation; disclose unavailable checks and report skipped comparisons separately from passing ones.

Open PRs against `main`. Use a clear commit title, keep changes focused and include reproduction details for behavior changes. Generated fixtures, model weights, raw validation packages and local working environments must stay outside git. This documentation refresh does not authorize uploading evidence packages.

Contributions use the workspace's declared [Apache-2.0 license](LICENSE). Be respectful in discussion; do not post private prompts, access tokens or vulnerability details. See [support](SUPPORT.md) and [security](SECURITY.md).
