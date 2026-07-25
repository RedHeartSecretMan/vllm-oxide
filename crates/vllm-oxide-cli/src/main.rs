//! `vllm-oxide` CLI — thin binary shell.
//!
//! Per ADR-0004 the CLI is intentionally minimal: parse argv, call
//! `LLM::generate`, print the completion. Lands in a downstream ticket once
//! the lib's `LLM::generate` API exists. Kept as a stub so the workspace
//! member compiles independently.

fn main() {
    eprintln!(
        "vllm-oxide-cli: stub. The LLM::generate API lands in a downstream ticket; \
         see issue #13 and the v0.1 spec (#12)."
    );
    std::process::exit(1);
}
