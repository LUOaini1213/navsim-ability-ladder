# Controlled-study methods: verified design, driving-score results pending

This is a methods note for the eventual manuscript, not a completed paper or
publication claim. Results must come from complete audited score archives; training
losses and smoke fixtures are not substitutes. Final source/measurement verification
is still required under `COMPLETION_REQUIREMENTS.md`.

## Cohort and split

The feature archives contain 103,288 navtrain and 12,146 navtest scenes. Current
controlled models use city-stratified, original-driving-session-disjoint navtrain
partitions: 83,099 training scenes in 130 sessions, 11,074 validation in 16 sessions
and 9,115 confirmation in 16 sessions. Segmented logs are not treated as independent
drivers: the original-session identifier removes the final numeric start/end suffix.
All 44 navtest sessions are disjoint from the navtrain sessions.

This protocol differs from the archived ladder's log-level validation scheme.
The historical navtest already informed the hypotheses; it remains exploratory.
The new confirmation is excluded from new-model optimization, normalization and
validation diagnostics, but was available to historical models. The trainer loads
the compressed source archive, including target arrays, then indexes training and
validation only; it does not use confirmation targets for either operation.
The completed input-control audit enumerated the unchanged official scene filters
on the available 1,192 navtrain and 136 navtest logs: all 103,288/12,146 selected
raw scenes have feature records, with zero missing listed logs, selected tokens
or selected-scene feature records (`results/audit_v2/input_controls.json`). This
is complete coverage of those selected cohorts, not all frames in the underlying
dataset or a reconstruction of historical extraction/fallback reasons. The cache
controller binds and explicitly reuses this earlier audit; it was not newly rerun
or retrospectively frozen before historical training.

## Current privileged inputs and target

Every core condition uses eight ego-status features and 20 centerline samples
over a 60 m lookahead, represented by local position/orientation (80 channels).
The object tensor contains the nearest eight selected current objects within 60 m:
local x/y, absolute velocity expressed in the ego basis, length/width, heading
cosine/sine, vehicle/pedestrian indicators and a validity mask. The selected
categories are `vehicle`, `pedestrian`, `bicycle` and `generic_object`; the latter
is not in the upstream `AGENT_TYPES` set. The feature builder's historical
`DYNAMIC` name does not establish that all included objects are recognized moving
actors. Bicycle/generic objects activate neither type indicator, so the tensor
does not distinguish those two categories explicitly. Padding is explicitly masked.
These are privileged annotations and map routes, not a learned sensor stack.

Outputs are eight relative x/y/heading poses at 0.5 s intervals, spanning four
seconds. The training objective is the coordinate-wise mean absolute error over
24 values mixing metres and radians. It is not an angular geodesic loss or a
calibrated safety objective. Diagnostics separately report coordinate L1 in m,
raw-heading L1 in rad, ADE and FDE in m. Progress-only ladder losses have different
units and must not be numerically equated with this mixed pose loss.

## Six-condition mechanism matrix

| condition | dynamic branch | independent lead summary |
|---|---|---|
| M0 | none, head ignores all objects | ignored |
| M1 | sorted real eight-object concat, including mask | ignored |
| M2 | matched cross-session donor concat | ignored |
| M3 | none | current bumper gap and forward lead speed |
| M4 | real objects, masked Deep Sets | ignored |
| M5 | same donors as M2, masked Deep Sets | ignored |

The shared pose head has three 512-wide ReLU hidden layers. Set conditions map
ten object channels through two 32-wide ReLU layers and concatenate masked mean
and max summaries with the 88-channel ego/centerline context. Empty sets produce
zero summaries; padded embeddings are excluded. This uses an established set
encoder and makes no architecture-novelty claim. Raw concatenation and the set
model do not have identical effective capacity despite a total-parameter gap below
the registered 10% tolerance.

Context normalization is fitted only on selected training scenes. Object means/
standard deviations are fitted to valid training objects, pooled over slots, for
the first six continuous channels. Heading/type/mask channels remain unscaled;
all padded continuous/categorical entries are forced to zero after transformation.
Inactive branches are ignored by the model rather than accidentally exposed as
normalized nonzero constants. All full-data conditions use the same normalizer.

## Donor controls and order diagnostic

Base donors stay in their partition and come from a different original session.
They match current driving command, speed bins [2,5,10,20] m/s and valid-object
count bins [1,3,6]. When needed, matching relaxes count, then speed, then command;
each level is recorded. A donor session is sampled before a donor scene.
No future pose or future actor annotation is a matching variable.

The input-only audit found substantial cross-city mixing. These base controls
therefore perturb both scene alignment and geographical distributions. They are
not clean physical counterfactuals, nor a direct causal test of traffic-interaction
use. A key alignment interpretation requires the separate same-city control or
must be withheld. Validation/confirmation each have only one Singapore session,
so strict same-city cross-session controls cannot support that city's comparison.

The separately frozen city input audit passes both strict city and different-
session invariants (`results/audit_v2/city_inputs.json`). Its training cohort stays
at 83,099 scenes/130 sessions, so the full-data real-input M1/M4 anchors and their
training-only normalizers can be reused. Common validation coverage is 10,312
scenes/15 sessions (762 Singapore scenes excluded); confirmation is 8,379/15
(736 excluded); exploratory navtest remains 12,146/44. The base protocol, donor
mapping and complete core artifacts are preserved, rather than repaired in place.
Only M2/M5 donor conditions need six auxiliary fits (three paired seeds). Their
matched-coverage scoring and six-model completion audit are still pending; input
invariants alone do not establish a clean alignment result. City control still
does not enforce route/actor compatibility or constitute a physical counterfactual.

Reversing the same scene's object order is a distinct diagnostic: it preserves
the object set and tests encoding sensitivity. Its maximum pose difference is
recorded at inference without retraining; it is not interchangeable with donors.

## Optimization, checkpointing and resources

Seeds 0/1/2 are paired across all six conditions. Adam uses learning rate 0.001,
batch size 256, no weight decay/dropout and the last checkpoint. Every full-data
run has 13,000 updates (40 full-data virtual epochs); validation losses are diagnostic,
not a checkpoint-selection rule. Frozen training-code/protocol/donor hashes prevent
in-place source changes from silently entering resumed models.

All 18 actual full-budget core runs are complete and passed the artifact audit
(`results/audit_v2/core_manifest.json`): each has 13,000 updates and the same full-
data normalizer. The measured largest/smallest parameter-count ratio is 1.0772576.
The separate training-diagnostic table/accepted convergence PNG and SVG concern
imitation losses, optimization and network-only timing; they do not establish
PDMS effects, statistical superiority or deployment safety.

Checkpoints include model, Adam state, RNG state, shuffled sampler order/position
and history. The exact-next-update unit test passes. An actual process was also
interrupted after update 325 and resumed to 1,000 smoke updates; its final model,
Adam, RNG, sampler and validation history exactly matched an uninterrupted run
(`results/audit_v2/process_recovery.json`). This exercise is not a full-budget
scientific model. Ineligible smoke runs have
separate tags/roots and cannot enter paper evaluation or confirmation.

Execution is serial with two CPU threads, below-normal child priority and a
free-memory guard. Reported latency is the warmed, single-scene neural forward
pass averaged over 200 repetitions, excluding map extraction, feature building,
metric simulation, I/O and sensor processing. Timing on a shared CPU is diagnostic,
not an end-to-end real-time or deployment claim.

## Evaluation and uncertainty

Strict prediction export records complete token/log coverage, finite poses,
checkpoint/normalizer/settings/feature/donor lineage and zero cached-inference
fallbacks. This does not prove zero historical live fallbacks. Real input parity
uses history-only scenes. The actual audit covers 16 deterministic navtest inputs
and four real-input conditions at seed 0 (M0/M1/M3/M4), 64 comparisons with maximum
pose error 7.63e-6 and no fallback in that sample. It is not an exhaustive live
audit of all scenes/seeds; donor controls have their separate frozen cached-input
semantics. The raw-cohort audit finds feature records for every selected official
training/test token, without establishing exhaustive historical fallback reasons.
Confirmation export requires all 18 full-budget core artifacts. Metric caches for
new navtrain partitions are separate from historical navtest caches.

Scoring calls the unchanged NAVSIM v1 PDM function and default configuration,
recording PDMS, NC, DAC, TTC, EP, comfort and driving-direction compliance. The
local v1.1 default uses progress/TTC/comfort weights 5/5/2 and a zero DDC weight;
DDC is reported diagnostically but does not change this configured PDMS. Cache
or score exceptions are explicit; missing scenes are not silently skipped. SQLite
recovery retains per-token work. Attempted failures and final invalid scores are
different quantities and must both be disclosed.

The three primary contrasts are M1-M0, M4-M1 and M4-M5. Report all paired seed
effects and their scene-weighted average. A whole-session paired bootstrap
resamples original sessions and divides score sums by scene counts; log clustering
is sensitivity analysis. The primary three intervals use Bonferroni tail allocation;
the percentile coverage is approximate. Intervals condition on the fitted seeds,
not uncertainty over newly trained models, and are not scores of an ensemble
trajectory. Subgroups and secondary contrasts are exploratory. Fewer than two
sessions gives no estimable population interval, not a zero-width one.

Input-defined strata and thresholds are frozen. The potential-crossing stratum
is a constant-velocity straight-axis proxy, not a map-aware collision/cut-in label.
All groups overlap. Diagnostic cases are outcome-conditioned samples inside fixed
positive/negative/neutral PDMS bins, hash-ranked with one case per original session
per contrast; missing bins are not filled by changing thresholds. All three seeds
are shown. Logged future ego plans in figures are evaluation-only; plan overlays
do not depict the official simulated rollout or establish collision validity.

Learning curves, epoch sensitivity, extra seeds, stronger donors and NAVSIM
benchmark-v2 extensions are governed by the frozen follow-up rules. Their absence
needs a gate/disclosure, not an implied positive result. NAVSIM v1 is nonreactive;
even benchmark-v2 reactive traffic would not by itself establish repeated ego
replanning or full closed-loop deployment safety.

## Conditional execution rules and claim limits

The numeric rules in `protocol/followup_policy.json` were bound after core
training started and before complete controlled PDMS comparisons. They allocate
resources and constrain interpretation; they are not a claim that the original
prose plan numerically preregistered them, nor a global significance guarantee.

- Any primary validation mean with absolute PDMS difference at least 0.01,
  including a negative or seed-unstable effect, triggers the M1/M4 nested 10/30%
  data follow-up: 12 fixed-13,000-update and 12 equal-epoch fits. The six 100%
  core anchors are reused, not independently replicated in the epoch panel.
- A primary contrast having paired seed effects at or below -0.01 and at or above
  +0.01 in validation or confirmation requires a separately frozen five-seed
  follow-up for any key claim based on it; otherwise that claim is withheld.
- The same-city execution rule was bound at 08:39:33 UTC on 2026-10-09, before
  complete core reports. An absolute validation M1-M2 or M4-M5 mean at least 0.01
  triggers all six auxiliary fits, including negative effects; a non-trigger
  withholds a clean-alignment interpretation. It is an additional prospective
  execution rule, not an original-plan numerical preregistration claim.
- Benchmark-v2 feasibility is triggered only when the same primary contrast in
  both validation and confirmation reaches the 0.01 practical threshold with
  the same direction, at least two of three seeds sharing it in each split, and
  both adjusted session intervals excluding zero in that direction. A triggered
  isolated M0/M1/M4 three-seed feasibility attempt is not already a v2 result.

These decisions must be recomputed from complete, hash-bound actual score
archives, not from loss tables, layout fixtures or edited summary means. At this
update the complete core score reports and gate decisions still do not exist.
Auxiliary city/curve analyses use paired original-session intervals conditional
on fitted seeds, and explicitly label their multiplicity sensitivities and
outcome-gated/exploratory scope. They do not extend the three core-primary
comparisons into an unconditional confirmatory family. An unsupported mechanism
remains an admissible technical-report result after the applicable work is
actually completed and disclosed.
