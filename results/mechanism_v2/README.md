# Controlled mechanism study (`mechanism_v2`): infrastructure, no results

**Status, 2026-11-03: infrastructure only, no results.** The controlled mechanism,
learning-curve, same-city and added-seed studies (`results/mechanism_v2/`) never
produced a PDMS result. Core training finished (18 models); the evaluation, cache and
scoring controllers were all stopped on 2026-10-09 and nothing has run since. Every
"live", "waiting" or "currently" in the docs and receipts describes the state recorded on 2026-10-09,
not a running process. Status JSON files such as `cache_resume_status.json`
(`waiting_for_memory`) and `scoring_status.json` (`waiting_for_cache_lock`) are the
last states written before those processes stopped. Workspace paths in tracked text
files read `<NAVSIM_WORKSPACE>`; the gated controllers re-hash the paths their
receipts name, so a new run has to start from a fresh attempt rather than resume
these. The zipped source snapshots under `results/repro_snapshots/` keep their
original bytes, including the original local paths.

What is here: the frozen protocols (`protocol/`, `city_protocol/`, `seed_protocol/`), the
training diagnostics of the 18 finished core models (`training_diagnostics/`,
`figures/training/`), a smoke test of the scoring path (`smoke_score/`, marked `eligible_for_paper: false`), and
the receipts of every attempt (`run_attempts/`). What is not here: any navtest PDMS of a
mechanism, learning-curve, same-city or added-seed model. Nothing in this folder supports a
claim about those questions; the execution record is
[`docs/EXECUTION_STATUS.md`](../../docs/EXECUTION_STATUS.md).
