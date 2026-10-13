# Issue and pull request workflow

[Documentation index](../README.md) · [Contributing](../../CONTRIBUTING.md)

GitHub Issues track bugs, proposals and implementation tickets for
`RedHeartSecretMan/vllm-oxide`. Use the `gh` CLI for tracker operations. Pull
requests contain proposed changes; feature scope and dependency-ordered work
remain in issues.

## Read before writing

Read the issue body, comments, labels, assignees and dependencies before
implementing a ticket. A `ready-for-agent` label describes a specified scope;
open blockers still prevent execution. The [triage guide](triage-labels.md)
defines the five workflow labels.

```bash
gh issue view 42 --repo RedHeartSecretMan/vllm-oxide --comments
gh issue list --repo RedHeartSecretMan/vllm-oxide --state open \
    --limit 100 --json number,title,labels,assignees
gh pr view 42 --repo RedHeartSecretMan/vllm-oxide --comments
gh pr diff 42 --repo RedHeartSecretMan/vllm-oxide
```

Issues and PRs share a number space. Resolve a reference such as `#42` to its
actual type before choosing `gh issue` or `gh pr` operations. Use explicit
`--repo` outside the repository; inside a clone, `gh` can infer the remote.

## Create and update

Write multiline issue descriptions, PR descriptions and comments to a file,
then pass `--body-file`. A quoted heredoc preserves literal Markdown, backticks
and shell examples without evaluating their contents.

```bash
issue_body_file=$(mktemp)
cat > "$issue_body_file" <<'BODY'
## Problem

Describe the concrete trigger and observed behavior.

## Expected result

Describe the change and its acceptance criteria.
BODY
gh issue create --repo RedHeartSecretMan/vllm-oxide \
    --title 'Describe the problem' --label needs-triage \
    --body-file "$issue_body_file"
rm "$issue_body_file"
```

Use the same `--body-file` pattern with `gh issue comment`, `gh issue edit`,
`gh pr create` and `gh pr comment`. Add or remove labels through `gh issue edit`
or `gh pr edit`. Keep comments factual: include decisions, concrete results and
source links rather than claiming a check that was not run.

When resolving an issue, record the result and implementing PR or commit in a
comment, then close it with the appropriate reason. `completed` means its scope
was delivered; `not planned` covers declined or superseded work. A draft PR
closed without merging should explain whether its change was integrated
elsewhere, replaced or not adopted.

## Dependency-ordered work

Large approved work can use a single overview issue and child implementation
tickets. Existing planning issues use `wayfinder:map` and `wayfinder:<type>`
labels. Keep those historical relationships and their approved scope intact.

- Link child tickets using GitHub sub-issues. If that facility is unavailable,
  use a task list in the overview and `Part of #<overview>` in each child.
- Represent blockers with GitHub's native issue dependencies. A text
  `Blocked by: #<number>` line is a fallback when the native facility is unavailable.
- Inspect the overview's open children and their live blockers before selecting
  work. A runnable ticket is specified, unassigned and has no open blocker;
  preserve the overview's ordering when choosing between runnable tickets.
- Claim the selected ticket with `gh issue edit <number> --add-assignee @me`.
  Use `feat/t<NN>-<slug>` for its implementation branch.
- When complete, link the actual result, close the ticket and update the
  overview with the decision or result pointer.

The native dependency endpoint takes the blocker's numeric database `id`,
which differs from the visible issue number and GraphQL `node_id`:

```bash
blocker_id=$(gh api repos/RedHeartSecretMan/vllm-oxide/issues/29 --jq .id)
gh api --method POST \
    repos/RedHeartSecretMan/vllm-oxide/issues/30/dependencies/blocked_by \
    -F issue_id="$blocker_id"
gh api repos/RedHeartSecretMan/vllm-oxide/issues/30 \
    --jq '.issue_dependencies_summary.blocked_by'
```

The summary counts open blockers. The example adds “issue 30 is blocked by issue
29”; substitute the actual approved ticket pair before performing a write.
Closing every blocker makes that dependency gate clear. Updating documentation
or cleaning up completed work does not itself authorize changing an approved
issue graph.
