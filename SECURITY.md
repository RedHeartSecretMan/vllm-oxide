# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest 0.2.x release | Yes, best effort |
| 0.1.x and older | No; upgrade to 0.2.x |

Security fixes target the latest published 0.2.x release. This is a community
project with no guaranteed response or maintenance service level.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Use [GitHub Security Advisories](https://github.com/RedHeartSecretMan/vllm-oxide/security/advisories/new)
to report privately. Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Affected version or commit, platform and relevant model source

Reports are reviewed on a best-effort basis. Please keep vulnerability details
private until a fix or disclosure plan has been agreed.

## Scope

vllm-oxide is a single-process inference engine. Security-relevant areas include:

- Weight loading (parsing untrusted safetensors files)
- CUDA FFI boundaries (memory safety at the Rust to CUDA seam)
- Network access during Hub model download

Out of scope: model output content safety, prompt injection.

Use model files from trusted sources and pin their revision when reproducing a
problem. Do not include access tokens, private model weights or sensitive prompts
in reports. Ordinary bugs and usage questions belong in the channels listed in
[SUPPORT.md](SUPPORT.md).
