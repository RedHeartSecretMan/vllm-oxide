# Getting Help

Start with the [README](README.md), [documentation index](docs/README.md) and
[v0.2.0 release notes](docs/releases/v0.2.0.md) for installation, examples and
supported behavior. This repository uses GitHub Issues for public questions and
reports; there is no separate chat or support service.

| You need to… | Use |
| --- | --- |
| Ask a usage or development question | Search [GitHub Issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues), then open a [question](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new?labels=question&title=%5BQuestion%5D%20) if needed |
| Report an error or unexpected behavior | [Bug report](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new?template=bug_report.yml) |
| Suggest a capability | [Feature request](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new?template=feature_request.yml) |
| Report unclear or incorrect documentation | [Documentation issue](https://github.com/RedHeartSecretMan/vllm-oxide/issues/new?template=documentation.yml), or submit a small correction directly |
| Report a security vulnerability | Follow the private reporting instructions in [SECURITY.md](SECURITY.md) |
| Contribute a change | [CONTRIBUTING.md](CONTRIBUTING.md) |

For an inference problem, include the version or commit, minimal command or Rust
example, model ID and revision, operating system, GPU/driver/CUDA details,
generation settings, expected result and actual result. For CPU or documentation
issues, GPU fields can be marked as not applicable. Short, relevant logs usually
help more than a large archive.

Remove tokens, private prompts and other sensitive information from public
reports. You do not need to upload model weights or raw acceptance evidence
packages to request help. Responses and fixes are best effort, with no promised
turnaround time.
