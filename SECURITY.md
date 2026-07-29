# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Use [GitHub Security Advisories](https://github.com/RedHeartSecretMan/vllm-oxide/security/advisories/new)
to report privately. Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You can expect an initial response within 72 hours.

## Scope

vllm-oxide is a single-process inference engine. Security-relevant areas include:

- Weight loading (parsing untrusted safetensors files)
- CUDA FFI boundaries (memory safety at the Rust to CUDA seam)
- Network access during Hub model download

Out of scope: model output content safety, prompt injection.
