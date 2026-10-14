# Contributing to vllm-oxide

You can contribute documentation, code, tests, or a reproducible issue report in English or Chinese. Many tasks need no GPU.

Use this archive to reproduce v0.1.0 behavior. Start new contributions from the current `main` branch and follow its [contribution guide](https://github.com/RedHeartSecretMan/vllm-oxide/blob/main/CONTRIBUTING.md).

## Choose a change

- For a small documentation correction, update the relevant guide and keep the two README languages aligned.
- For a bug, search [existing issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues) and include the exact source commit, command and expected/actual behavior.
- For a feature or API change, explain the use case and proposed behavior before a large implementation.
- For a ready ticket, follow its acceptance criteria and the [issue workflow](docs/agents/issue-tracker.md).

## Validate and submit

Use the [development guide](docs/development.md) for version-correct checks. Explain what changed and why, which checks actually ran, and any compatibility impact. CPU-only changes need no model download or GPU. Inference changes require relevant GPU validation; disclose unavailable checks and report skipped comparisons separately from passing ones.

Open PRs against `main`. Use a clear commit title, keep changes focused and include reproduction details for behavior changes. Generated fixtures, model weights, raw validation packages and local working environments must stay outside git. Upload evidence packages only when explicitly requested.

Contributions use the workspace's declared [Apache-2.0 license](LICENSE). Be respectful in discussion; do not post private prompts, access tokens or vulnerability details. See [support](SUPPORT.md) and [security](SECURITY.md).
