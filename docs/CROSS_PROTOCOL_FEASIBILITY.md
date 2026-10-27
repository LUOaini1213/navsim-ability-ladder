# Conditional NAVSIM benchmark-v2 feasibility

This is a prospective feasibility checklist, not an executed v2 experiment or a
change to the frozen core-study protocol. The core-study label `mechanism_v2`
does not mean NAVSIM benchmark v2. No v2 result exists at this update.

## Primary-source boundary checked 2026-10-09

The [official v2.2 metrics specification](https://raw.githubusercontent.com/autonomousvision/navsim/v2.2/docs/metrics.md)
distinguishes v1 PDMS from EPDMS (additional direction/traffic-light multipliers,
lane/extended-comfort metrics and human-violation filtering). Its two-stage
aggregation evaluates precomputed follow-up states and weights them by the
first-stage endpoint. Neither the one-stage v2 scorer nor this aggregation is
full interactive closed-loop deployment validation. The
[official repository](https://github.com/autonomousvision/navsim) also records a
later 2025-09-29 human-filter bug fix. Thus a v2.2 branch name alone is insufficient
version provenance; pin and disclose an exact commit and relevant fix behavior.

## Gates before representative model execution

1. Save the registered validation/confirmation cross-protocol gate and recompute
   it from the complete hash-bound core score archives. No result-driven threshold
   adjustment or conversion of imitation loss into a PDMS finding.
2. Create a separate pinned checkout/environment, never upgrade the user's v1.1
   devkit in place. Record dependency and local-patch hashes and inspect the
   human-filter behavior. Existing v1 caches must not be loaded as v2 caches.
3. Audit raw log/token/original-session and source-data overlap, and separate
   one-stage `navtest` evaluation from genuine two-stage dataset evaluation.
   Existing test logs alone do not establish availability of follow-up states.
4. Test history-only structured ego/map/object feature extraction against raw
   v2 and synthetic follow-up states, including map coordinates, object velocity,
   categories/masks and future exclusion. Fail rather than falling back silently.
5. Run an isolated no-sensor cache/score smoke; compare to the pinned official
   scorer and audit complete submetric/human-filter outputs. Record peak RAM,
   disk growth and scenes/hour under the same serial two-thread resource policy.
6. If feasible, run the registered representative M0/M1/M4 three-seed comparison
   with separate caches and frozen coverage. Report one/two-stage status and
   differences in metrics without comparing unlike aggregate scores directly.
7. If unsupported, retain the actual failed command/exception/dependency or
   measured resource evidence and disclose the exact limitation in the manuscript.
   A hypothetical CPU limitation is not a measured feasibility failure.

The [official installation guide](https://raw.githubusercontent.com/autonomousvision/navsim/v2.2/docs/install.md)
routes two-stage evaluation through separate downloads. Do not download large
sensor datasets merely to make a planning-only extension appear complete; first
verify the structured-input smoke and measured resource/storage feasibility.

## Paper-grounded adapter preflight (not executed)

The [arXiv v3 main paper](https://arxiv.org/abs/2506.04218v3) was read and visually
checked on 2026-10-09; the separately linked supplement was inaccessible behind
a browser-verification challenge. Review scope and version-specific limitations
are recorded in `CLOSEST_PRIOR_REVIEW.md`. The following are our implementation
checks inferred from that protocol, not claims that an adapter already works:

- Bind each Stage-2 observation to its synthetic pose, heading, ego motion
  history and parent Stage-1 token. Audit every ego/map/object transformation
  with known rotations and actual raw examples; do not substitute original
  expert ego history, leak future labels, or synthesize inputs from the model's
  predicted future without the pinned official state-construction semantics.
- Independently trace the official Stage-1 controller's reached endpoint and
  the Stage-2 start points in the same frame. Check normalized proximity weights,
  kernel/config identity and two-stage product against official outputs, not
  a hand-coded weighting of the last requested waypoint or a simple stage mean.
- Keep all nine EPDMS components, human-filter flags and configuration/NumPy
  versions in the new receipts. A reduced correlation-study metric or the seven
  v1 score columns is not the official full benchmark metric. Audit the pinned
  implementation's actual human-filter fix rather than assuming paper equations
  resolve later code changes.
- Enumerate actual initial/synthetic coverage and missing parents before
  aggregation. Separate correlation-study cohorts, navhard and one-stage
  navtest in reports; never fill missing follow-up states with expert-aligned
  observations or compare the old privileged MLP scores to the paper's sensor
  leaderboard as matched results. Record measured feasibility before any claim
  of CPU infeasibility or sensor-free support.

These checks refine stages 3-5 above without relaxing the existing actual-score
gate, memory headroom or representative-model requirements. No large data
download, environment upgrade, adapter execution or v2 model run occurred.

## Cross-benchmark correlation is not this migration's proof

The [cross-benchmark correlation preprint](https://arxiv.org/abs/2605.00066v1)
was separately read in full with its table arithmetic checked. Its eight complete
method-level pairs concern PDMS and published Bench2Drive scores, not this
project's models or eight full two-stage EPDMS results. CARLA-v2 and Bench2Drive
score conventions are separately labelled in its tables. Do not use its
correlation or simplified proxy to waive the actual state/coverage/scorer
checks above, change our registered official metric, or claim external validation.
Its underlying cited model configurations were not independently reproduced.

## Current status

Official-source checklist and bounded primary-paper review complete; gate,
isolated environment, overlap audit, synthetic-state adapter smoke and measured
v2 feasibility are all pending. This is not a separate-supplement review.
This file does not waive a triggered experiment or authorize an external submission.
