# Research audit — 2026-10-09

## Completed evidence

The local workspace contains all 27 trained cells and 56 navtest evaluations.
The repository already archives their per-scene scores and training metadata.
No training process was running when this audit began.

Direct checks of the frozen workspace features found zero overlapping tokens
and zero overlapping log names between navtrain and navtest. All 27 cells use
the same validation-log set and the last-epoch checkpoint policy. The navtest
feature archive contains 12,146 unique tokens.

## Statistical revision

Run `python analysis/analyze_clustered.py` to reproduce
`results/navtest/clustered.md` and `clustered.json` from the committed scores.
The paired percentile bootstrap samples 136 complete logs, preserving dependence
between scenes within each log. It keeps the scene-weighted mean by dividing
resampled score sums by resampled scene counts. Learned comparisons average the
three fitted seeds before resampling. These intervals condition on those seeds;
three seeds do not establish general training-run uncertainty. Contrasts are
exploratory and have no multiplicity correction.

| Finding | Delta PDMS | 95% log-cluster interval |
|---|---:|---|
| Map added to pose MLP without boxes | +0.1811 | [+0.1621, +0.1999] |
| Boxes added to pose MLP without map | +0.0174 | [+0.0022, +0.0320] |
| Boxes added to pose MLP with map | -0.0407 | [-0.0575, -0.0249] |
| Map x boxes interaction | -0.0581 | [-0.0704, -0.0465] |
| Learned speed on rule path vs rule speed | +0.0313 | [+0.0249, +0.0382] |
| Rule path + learned speed vs full pose, map inputs | -0.0060 | [-0.0176, +0.0061] |
| Dropout 0.2 vs baseline, seed 0 | -0.0124 | [-0.0297, +0.0032] |

The current strongest claim is an information-by-model interaction under this
specific feature representation and training budget. More privileged inputs did
not consistently improve these small imitation models. This does not imply that
obstacle information is intrinsically unhelpful: nearest-eight sorting, raw
concatenation, optimization and imitation targets are possible explanations.

The full-data pose MLP has mean PDMS about 0.823; the rule-path speed model about
0.817. The earlier small-data claim that path decomposition wins does not
generalize. A CI containing zero establishes neither superiority nor equivalence.

The dropout result no longer supports a robust population-level deterioration
claim. Mild weight decay improves both validation fit and PDMS in the one tested
seed; the regularisation sweep is exploratory, not a multi-seed conclusion.

## Implementation limitations affecting paper claims

1. `LadderMLPAgent` calls `scene_blocks` even for ego-only models. That helper
   accesses route geometry and returns no features when the route is unusable;
   the agent then falls back to a kinematic rule. Therefore ego-only MLP inputs
   are blind, but the execution path still has a map-dependent gate. Existing
   scores require a fallback-frequency audit before describing these variants
   as strictly map-independent systems.
2. Broad exception handling silently falls back to the kinematic rule. Zero
   failed scoring jobs does not establish zero model failures or fallbacks.
   Add explicit counters and audit representative scenes before additional runs.
3. Learned-path retiming clips progress to the predicted polyline length. The
   path/speed table is a useful intervention, but not a clean independent 2x2:
   predicted path extent retains information from the predicted speed profile.
4. Changing from 563 scenes to navtest also increased training data and changed
   splits. The reversal cannot be attributed to test sample size alone. Matched
   training-size experiments are needed to isolate sample efficiency.
5. Rule-map and rule-box comparisons change both information and control law;
   their sevenfold ratio is descriptive, not a causal information-value ratio.
6. NAVSIM v1 is non-reactive simulation-based evaluation. Use PDMS or
   simulation-based performance, not claims of interactive closed-loop safety.
7. Map/route features are privileged. Published sensor-based models are contextual
   references with different inputs, not controlled competitors.

## Next experiments, in order

- Instrument fallbacks and verify ego-only inputs bypass unnecessary map access;
  preserve existing runs as versioned historical results.
- Check obstacle-coordinate/velocity conventions and feature parity on real
  scenes. Compare padded raw concatenation with a masked permutation-invariant
  encoder at matched training budget and report parameter counts.
- Add a random-feature or shuffled-obstacle control to distinguish useful
  obstacle signal from extra input dimensions; freeze its protocol on validation.
- Train matched-size learning curves using disjoint whole-log subsets and three
  seeds to separate training-data effects from evaluation-split effects.
- Evaluate new choices on validation before a frozen final test protocol. The
  current navtest has already informed hypothesis development.

These are outstanding research tasks. The present artifact is an auditable
analysis report, not a completed or accepted paper.

## Implementation follow-through

See [the execution record](EXECUTION_STATUS.md). The corrected canonical agent
now builds only requested inputs and exposes every model/fallback diagnostic.
The v1 runtime checkout and historical checkpoints/scores were not overwritten.

The formal feature audit covers all 27 archived checkpoints and both complete
feature archives. Their stored normalization matches training-only recomputation
exactly; none of their cached-navtest outputs is nonfinite. Real-scene parity is
sampled: 48 scenes across 24 logs, all passing, including history-only extraction
and four seed-0 live/cached pose-model comparisons per scene. This does not prove
zero historical fallbacks over the complete dataset.

An additional 128 score-path comparisons (32 scenes x four archived agents)
reproduce PDMS and subscores within 4.96e-7 using the unchanged official scorer.
See `results/audit_v2/audit.json`, `feature_ledger.jsonl`, and `score_path.json`.

The logs represent 44 original sessions. Session-level resampling gives
boxes-without-map +0.0174 [-0.0082,+0.0414], boxes-with-map -0.0407
[-0.0667,-0.0163], and interaction -0.0581 [-0.0707,-0.0460]. Accordingly,
the first effect is not robust to the more conservative clustering unit.

V2 splits are now session-disjoint and city-stratified: 83,099 training scenes
(130 sessions), 11,074 validation (16), and 9,115 confirmation (16). The original
navtest has 44 disjoint sessions. This differs from the historical log-level
validation scheme; v2 results must not be presented as an identical-protocol
rerun of v1. Historical models saw the new confirmation data, so it is not
project-pristine unseen data. The v2 trainer reads the complete compressed
archive but uses only training/validation indices for optimization/diagnostics;
confirmation targets are not used for either.
