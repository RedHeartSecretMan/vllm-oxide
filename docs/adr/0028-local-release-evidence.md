---
status: accepted
---

# Retain complete release evidence locally

The user explicitly instructed that the evidence bundle does not need to be
pushed to GitHub. Retain the complete original closure and the schema5
`manifest.json` / `goldens-v0.2.tar.gz` pair locally, independently verify them
with Python and Rust consumers, and render the source-bound release report.
Withdraw the proposed split transport; do not introduce schema6, omit raw
captures, rewrite provenance or claim that the new 747-owner confirmation has
passed. Accuracy, behavior, performance, both source-role CPU gates and fresh
independent reviews remain required for release acceptance.

GitHub's strictly-less-than-2-GiB asset limit applies only to optional manual
publication. Local packing and installation instead enforce a finite 2-TiB
compressed archive bound while retaining the original 16-MiB manifest,
50,000-artifact, 8-GiB-minus-one-byte per-file and 1-TiB extracted-total bounds.
Canonical gzip/USTAR, streaming length/hash verification, disk preflight,
complete semantic evaluation and atomic immutable installation are unchanged.
This explicitly supersedes ADR-0016's requirement to upload the two evidence
files for this acceptance run; its complete-evidence and schema5 safeguards
remain in force. Product publication under #47 remains a separate operation.

The executable capacity repair is limited to the Python and Rust
`release_transport` modules and their tests, with associated documentation and
Definition-index updates, in a separately frozen supervision/evaluation source
Y. The calibrated `2dc9445` source still derives measurement X only through the
approved numerical policy, its supervision digest pointer, the Definition index
and approval ADR. Y must descend from the fixed X, bind X's actual source and
binaries, preserve every protected measurement Git object and the reference and
mathematical evaluator, and pass complete CPU gates and full review before use.
This narrow byte-transport exception does not relax calibration-to-measurement
executable equivalence or the final evidence-only descendant rule. Calibration
owners cannot be reused as authoritative owners; the proposed independent scope
still needs its own approval. No registry or numerical budget changes are part
of this decision, and no golden tag, release or upload is authorized.
