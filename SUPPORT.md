# Getting help

This checkout documents historical `v0.2.0.rc.1`. New users should upgrade to [v0.2.0](https://github.com/RedHeartSecretMan/vllm-oxide/releases/tag/v0.2.0).

Start with the [README](README.md), [documentation index](docs/README.md) and
[RC release notes](docs/releases/v0.2.0-rc.1.md) for installation, examples and
supported behavior. This repository uses GitHub Issues for public questions and
reports. English and Chinese are both welcome. There is no separate chat,
discussion forum or support service.

| You need to… | Use |
| --- | --- |
| Ask a usage or development question | Search [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues), then open a [question](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new) if needed |
| Report an error or unexpected behavior | [Bug report](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new) |
| Suggest a capability | [Feature request](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new) |
| Report unclear or incorrect documentation | [Documentation issue](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new), or submit a small correction directly |
| Report a security vulnerability | Follow the private reporting instructions in [SECURITY.md](SECURITY.md) |
| Contribute a change | [Contribution guide](CONTRIBUTING.md) and [development guide](docs/development.md) |

For an inference problem, include the version or commit, minimal command or Rust
example, model ID and revision, operating system, GPU/driver/CUDA details,
generation settings, expected result and actual result. For CPU or documentation
issues, GPU fields can be marked as not applicable. Short, relevant logs usually
help more than a large archive.

Remove access tokens, private prompts and other sensitive information from
public reports. Model weights and raw validation packages are not needed to
request help. Maintained releases are listed in [SECURITY.md](SECURITY.md);
support for the current 0.2.x series is best effort, with no promised turnaround
time. An issue or PR label communicates workflow status, not a release date.
