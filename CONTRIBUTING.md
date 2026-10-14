# Contributing to vllm-oxide

You can contribute code, tests, documentation, or a reproducible issue report.
Many tasks need no GPU. Feel free to use English or Chinese in issues and pull
requests.

Follow the [community standards](CODE_OF_CONDUCT.md) in discussions and reviews.

Start with the [documentation index](docs/README.md) and the
[historical RC scope](docs/releases/v0.2.0-rc.1.en.md). For usage questions, see
[SUPPORT.md](SUPPORT.md); report vulnerabilities privately through
[SECURITY.md](SECURITY.md).

## Choose a contribution

- Send small documentation corrections directly as a PR. Check links and
  examples; a wording or link fix needs no GPU validation.
- For a bug fix, search [existing issues](https://github.com/RedHeartSecretMan/vllm-oxide/issues),
  include a minimal reproduction, and add a regression test when behavior changes.
- For a feature or API change, describe the use case and proposed behavior in an
  issue before a large implementation. Capabilities outside the current release
  scope require their own design and validation.
- For a ready ticket, follow its acceptance criteria and use
  `feat/t<NN>-<slug>` for the branch. The [triage guide](docs/agents/triage-labels.md)
  explains readiness; the [issue workflow](docs/agents/issue-tracker.md) explains
  dependencies.

## Develop and validate

Follow the [development guide](docs/development.md) to set up Rust 1.94.0 and the
locked Python environments. It explains the code conventions and which CPU or
GPU checks to run for your change.

This maintained RC has a CPU-only Rust 1.94.0/stable matrix, default/internal-golden tests and native dependency auditing. Modern contributions should target `main`, which also has Python and documentation automation. No runner setup,
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
