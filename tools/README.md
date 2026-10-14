# Repository tools

[Project overview](../README.md) · [Documentation](../docs/README.md) · [Development](../docs/development.md)

Use these utilities to check documentation and collect or inspect validation
evidence. The Python package is for development; the inference library and CLI
do not import it. Run commands from the repository root unless a linked guide
specifies another directory.

## Choose the task

| Tool | Task | Runtime and output |
|---|---|---|
| [`check_docs.py`](check_docs.py) | Check local links, reference links and heading anchors in Git-tracked Markdown | Python standard library, CPU, no network; prints errors and a document/error count; nonzero exit on errors |
| [`test_check_docs.py`](test_check_docs.py) | Regression tests for the documentation checker's parsing and path handling | Python `unittest`, CPU; runs in ordinary documentation CI |
| [`golden-gen/`](golden-gen/README.md) | Layered accuracy collection/evaluation, CPU adapter tests, fixed-workload performance and local evidence transport | Python 3.12 and locked dependencies; actual model capture/performance requires a prepared CUDA host |
| [`golden-gen/scripts/`](golden-gen/scripts/README.md) | Small BF16 operator probes and fixed historical layer localization | CPU or GPU depending on the script; diagnostic output, no acceptance marker |
| [`validate-release.sh`](validate-release.sh) | Retained historical schema-v4 orchestration | Fixed historical run paths and predecessor markers; `authoritative` and `publish` are disabled; not the current release command |

The Rust capture, benchmark and bundle-reader executables are documented in
[`crates/vllm-oxide-test`](../crates/vllm-oxide-test/README.md). The historical
[Candle experiment](../research/prototype-candle-stack/README.md) is separate
from both the supported engine and these validation utilities.

## Check documentation

```bash
python3 -m unittest discover -s tools -p test_check_docs.py
python3 tools/check_docs.py
```

`check_docs.py --root /absolute/path/to/checkout` selects another Git checkout.
The checker uses `git ls-files`, so newly created Markdown must be staged before
it is included in the normal check. It ignores remote URLs and code blocks;
it does not validate external website availability, execute examples or render
Markdown. The documentation workflow separately builds the public Rust API
documentation on a CPU runner.

## Validate code, models and evidence

Start with [golden-gen setup](golden-gen/README.md#setup-and-cpu-tests). Ordinary
GitHub CI uses CPU workers for Rust/Python tests, static checks and test-evidence
adapters. It needs no GPU. A local Linux or WSL2 GPU supplies the actual model,
public API/CLI and performance checks described in the
[validation guide](../docs/validation/README.md).

The current command families have separate responsibilities:

1. `python -m golden_gen.layered_cli`: frozen owner collection, replay,
   calibration observation, fault checks and authoritative L0/L1/L2 evaluation.
2. `python -m golden_gen.public_gpu_smoke`: production build, repeated public
   generation calls, mixed requests and the Quick Start CLI.
3. `python -m golden_gen.release_cli`: CPU evidence, fixed-workload performance,
   local bundle construction, verification and reports. Its optional golden
   publication action has separate authority; it does not release the product.
4. `python -m golden_gen`: retained schema-v4 fixture tooling, including a
   CPU-only synthetic dry run. It cannot substitute for the current protocol.

Detailed flags, prerequisites, source identities and stage outputs are in the
[package guide](golden-gen/README.md). The [maintainer release guide](../docs/releases/README.md)
covers product versioning, tags and GitHub Release content.

Use fresh evidence paths outside the reviewed source checkout and keep complete
raw evidence on durable local storage. `golden-gen/output/` is ignored working
state, must not be committed, and is not accepted release evidence. Model
weights, diagnostic captures and full evidence packages are not uploaded as part
of the current source releases. Earlier runs keep their original source
identities and verdicts.
