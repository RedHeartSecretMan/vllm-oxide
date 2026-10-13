# Contributing to vllm-oxide

Code, tests, documentation and reproducible issue reports are welcome. You can
contribute without a GPU. English and Chinese are both welcome in issues and
pull requests.

Follow the [community standards](CODE_OF_CONDUCT.md) in discussions and reviews.

Start with the [documentation index](docs/README.md) and the
[current release scope](docs/releases/v0.2.0-rc.1.en.md). For usage questions, see
[SUPPORT.md](SUPPORT.md); report vulnerabilities privately through
[SECURITY.md](SECURITY.md).

## Choose a contribution

- **Documentation:** send small corrections directly as a PR. Check links and
  examples; a wording or link fix needs no GPU validation.
- **Bug fix:** search [existing issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues),
  include a minimal reproduction, and add a regression test when behavior changes.
- **Feature or API change:** describe the use case and proposed behavior in an
  issue before a large implementation. Capabilities outside the current release
  scope require their own design and validation.
- **Ready ticket:** follow its acceptance criteria and use
  `feat/t<NN>-<slug>` for the branch. The [triage guide](docs/agents/triage-labels.md)
  explains readiness; the [issue workflow](docs/agents/issue-tracker.md) explains
  dependencies.

## Develop and validate

The [development guide](docs/development.md) covers Rust 1.94.0, local CPU checks,
the locked Python environments, code conventions and the distinction between
CPU CI and GPU validation. It also explains which checks apply to each change.

This historical checkout has automatic CPU-only Rust and dependency checks. Modern contributions should target `main`, which has the current test and documentation automation. No runner setup,
CUDA Toolkit or model download is needed for Rust and documentation checks.
Contributors changing inference behavior should report any relevant local GPU
validation and any checks they could not run.

## Submit a pull request

1. Work on a branch from `main` and keep the change focused.
2. Update the relevant tests and documentation when behavior changes. Keep the
   English and Chinese README aligned.
3. Run the checks relevant to the change and state their actual results.
4. Open a PR against `main`, describing the problem,
   resulting behavior, validation and any compatibility impact. Link the issue
   when there is one.

Use a clear commit title, such as `fix(sampler): handle an empty candidate set`
or `docs: explain local model loading`. The project uses
[Conventional Commits](https://www.conventionalcommits.org/); scopes are optional.

Keep model weights, generated fixtures, raw validation packages and local
working directories outside git. Include a concise validation summary in the
PR and coordinate access separately if a reviewer needs more evidence.

## License

Contributions are licensed under the project's [Apache-2.0 license](LICENSE).
