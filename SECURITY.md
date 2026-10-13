# Security policy

## Supported versions

| Version | Security support |
|---|---|
| Latest 0.2.x release | Best effort |
| 0.1.x and earlier | Historical; upgrade to 0.2.x |

The v0.1.0 documentation refresh does not establish a new support commitment. Security fixes target the latest published 0.2.x version, without a guaranteed response time.

## Report a vulnerability

Use [GitHub's private vulnerability report](https://github.com/RedHeartSecretMan/vllm-oxide/security/advisories/new). Do not open a public issue with vulnerability details. Include affected versions/commits, impact, minimal reproduction, model source and relevant platform details. Reports in English or Chinese are welcome.

Security-relevant areas include parsing model files, CUDA FFI memory boundaries and network access during Hub downloads. Model-output content safety and prompt injection are outside this engine's security scope.

Load model files from trusted sources and pin revisions for reproduction. Remove credentials, private model weights and sensitive prompts from reports. Ordinary bugs and questions use [SUPPORT.md](SUPPORT.md).
