# Triage labels

[Issue workflow](issue-tracker.md) · [Contributing](../../CONTRIBUTING.md)

The repository uses five canonical labels to describe an issue's workflow
status. Keep their names stable so people and agents share the same readiness
signal.

| Label | Meaning | What moves it forward |
|---|---|---|
| `needs-triage` | Awaiting initial evaluation | Confirm the problem, scope and appropriate next action |
| `needs-info` | Waiting for information from the reporter | Supply the requested reproduction, context or decision |
| `ready-for-agent` | Specified for agent implementation | Resolve open dependencies, then claim and implement the ticket |
| `ready-for-human` | Requires human implementation or a human-only action | Complete the identified action or refine the remaining scope |
| `wontfix` | The proposed work will not be pursued | Record the reason and close as not planned when appropriate |

Readiness includes a concrete expected result and acceptance criteria. A
`ready-for-agent` issue with an open native `blocked_by` dependency is not yet
runnable. A label also does not claim that work has passed validation.

Use one current triage status where possible and remove a stale status when
transitioning. Descriptive labels such as `bug`, `enhancement`, `documentation`
and `question` describe the kind of request separately. Release-note categories
in [`.github/release.yml`](../../.github/release.yml) also serve a different
purpose from readiness.

The [report forms](../../.github/ISSUE_TEMPLATE/) apply `needs-triage` initially.
Closed historical tickets and their dependency links remain records of the
scope and decisions that produced a release.
