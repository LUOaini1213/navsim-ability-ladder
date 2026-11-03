# Closest-prior full-text review

Review date: 2026-10-09. This is a comparison with selected primary sources,
not an exhaustive novelty certificate. No cited baseline was reproduced here.

## Driving on Memory

Source: [arXiv:2608.31029v1](https://arxiv.org/abs/2608.31029), a preprint.
Read all 23 PDF pages, including appendices. Visually checked Figure 2 and
Tables 2-6 against extracted text. The downloaded reference is read-only and
excluded from Git under `tmp/pdfs/`.

The paper replaces current cameras with pose-retrieved memories, while retaining
ego status. It reports NAVSIM v1/v2, Bench2Drive, RealEngine and HUGSIM results.
Importantly, it already includes HD-map-only inputs (Table 6), interaction groups
(Table 10), memory representation probes (Table 13), camera-memory fusion
(Table 14), and geographical controls (Table 7). Its short-horizon explanation
explicitly discusses pre-existing ego braking cues and conservative progress.
Appendix B.2 discloses different proposal-selection policies for some Bench2Drive
controls. These numbers are not a matched comparison with our small MLPs.

Our candidate distinction is **current-object alignment x encoder x data budget**
with a fixed ego/centerline context, separate lead summaries, controlled input
swaps, and paired session-cluster analysis. This paper does not establish that
our distinction is novel across all literature. Map strength, scenario breakdowns,
and adding NAVSIM v2 alone are insufficient novelty claims.

## PlanTF

Source: [Rethinking Imitation-based Planner for Autonomous Driving](https://arxiv.org/html/2309.10443v1).
Read the main text and supplementary result sections, including Tables I-VIII.

The study already relates input features to shortcut learning, uses a state
dropout encoder, tests augmentation/normalization, and separates trajectory
imitation from tracking dynamics. Its nuPlan protocol uses repeated closed-loop
planning, unlike our NAVSIM v1 scoring. Therefore “more inputs can hurt,” “an
encoder helps,” and open-/closed-loop disagreement are not standalone new ideas.
Our object-specific controls must distinguish an observed information effect
from extra dimensions, optimization, and input encoding. We do not reproduce
PlanTF or claim comparative superiority.

## Is Ego Status All You Need?

Source: [arXiv:2312.03031v2](https://arxiv.org/html/2312.03031v2).
Read its main text and appendices A-E in the primary HTML version.

The nuScenes study already tests camera corruption, ego-state perturbation,
map-task addition, straight/turn groups and convergence. It finds that input
effects can change with the evaluation metric and subgroup, and discusses
conservative plans. Appendix B aligns valid scene coverage and accounts for
yaw in collision checks; Appendix A identifies incorrect future labels in a
previous baseline. These are reasons to audit our extraction, not evidence
that the same bug exists here. Its inference-time corruptions are distinct from
our independently trained, cross-session donor controls. Raw sensor inputs,
nuScenes horizons and metric definitions differ from our privileged eight-object
NAVSIM experiment. Neither global perception redundancy nor safety equivalence
can follow from the comparison. Scenario effects alone are not a new contribution.

## PDM: centerline baseline and evaluation mismatch

Source: [Parting with Misconceptions about Learning-based Vehicle Motion Planning](https://arxiv.org/abs/2306.07962v2).
Read the 14-page primary PDF (main text pp. 1-8, followed by references);
visually checked p. 4, including Table 1. Also read all nine pages of the
[author-hosted supplement](https://danieldauner.github.io/assets/pdf/Dauner2023CORL_supplementary.pdf),
with a visual check of architecture/training and Table 4 on p. 6. It documents
separate nuPlan metric aggregation and training settings; our models do not copy
its history horizon, object forecasting or optimization schedule.

PDM-Open already conditions an MLP on an extracted centerline and ego history.
Table 1 separates centerline/history effects and open-/closed-loop performance;
Table 4 examines centerline resolution, length and model width. PDM-Closed also
uses object forecasting, proposals, controller simulation and scoring. Thus
centerline-only imitation is an established baseline, not a new architecture
here, and the paper cannot support removing environmental information from
general driving. Our four-second, one-query NAVSIM evaluation and present-state
eight-object branch are not a replication of its nuPlan evaluation. ADE reduction
cannot substitute for official PDMS evidence or deployment safety.

## NAVSIM: benchmark protocol and existing input/data ablations

Source: [NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking](https://arxiv.org/abs/2406.15349v2).
Read its main text pp. 1-10 in the 14-page primary PDF; visually checked Table 1
and Figure 4 on p. 7. Also read all 12 pages of the
[official NeurIPS supplement](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Supplemental-Datasets_and_Benchmarks_Track.pdf),
with a visual check of the 15-combination PlanCNN/data-fraction design on p. 6.
Its v1.1 update collects DDC without adding it to PDMS; local default weights were
checked independently against that description.
The arXiv paper revision `v2` is not the NAVSIM benchmark-version-2 protocol.

Section 4.1 already evaluates PlanCNN input combinations with three training
fractions when studying metric alignment. Section 4.2 reports three TransFuser
seeds, input-range/state ablations and component metrics. Therefore an input
ladder, data curve, three seeds or a subscore plot alone is not a distinct
contribution. The paper's sensor-input comparisons cannot be ranked against our
privileged object/map models as if inputs and training were matched. Its
non-reactive, one-query protocol omits replanning feedback and compounded errors;
the discussion explicitly distinguishes high PDMS from closed-loop performance.
Our candidate distinction needs aligned-vs-donor object controls, encoder
interaction, same-city qualification and uncertainty with independent-session
sampling, not a restatement of those generic ablations.

## Deep Sets: established architecture, bounded theoretical claim

Source: [Deep Sets](https://arxiv.org/abs/1703.06114v3).
Read the main text pp. 1-8, theory/architecture appendices A-C (pp. 12-19),
and point-cloud architecture appendix H (pp. 26-27). Visually checked the
theorem qualifications on p. 2. Other application-specific appendices are not
part of this selected review.

Shared element transforms and commutative aggregation are established, including
conditioned and max-pooling variants. The sum-decomposition theorem has stated
domain/continuity/set-size qualifications; it is not a performance guarantee for
our finite-width masked mean/max encoder. M4/M5 are Deep-Sets-style baselines,
not a new set architecture or theorem. Reversal and padding tests establish
implementation invariance on tested inputs, not superior interaction reasoning,
sample efficiency or safety. Such empirical effects require the planned controls.

## Set Transformer: attention architecture, not our mean/max encoder

Source: [ICML 2019 / PMLR 97](https://proceedings.mlr.press/v97/lee19d.html).
Read all ten publisher PDF pages (main pp. 1-8, references pp. 9-10) and
all seven pages of its publisher-hosted supplement. Visually inspected every
page, including main Figures 1-5/Tables 1-5, supplement Tables 1-10/Figure 1
and the theoretical statements. The two original PDFs/page hashes and scope
are retained in `run_attempts/2026-10-09_set_transformer_review.json`. This is
not author-code execution, numerical replication or independent proof verification.

SAB/ISAB provide permutation-equivariant attention encoders; learned-seed PMA
provides invariant pooling. ISAB's O(nm) attention cost is linear in input
size only with fixed inducing count m; full SAB has O(n^2) cost. PMA's
trainable seed vectors are architectural parameters, not independent training
random seeds. The supplement's universality construction uses sufficient
feed-forward capacity and a sum-pooling construction, not a finite-network
performance guarantee. Main Table 1's max pooling and Table 4's larger-set
ISAB+pooling results preclude a universal attention-dominance claim. Supplement
runtime is a Tesla-P40 dummy-set benchmark, not CPU planning latency. Some
main/supplement character-counting values differ; no checkpoint explanation
or numerical reconciliation is inferred.

Our inference, checked against the current read-only `MechanismModel`: M4/M5
use a shared row-wise MLP with masked mean/max, not SAB, ISAB or PMA. They
must remain described as Deep-Sets-style baselines, not Set Transformers or
a newly invented relational architecture. Reversal/padding tests do not prove
learned interaction reasoning. M4-M1 changes the representation/encoder as
a bundle; even a supported score difference would not isolate permutation
invariance alone as its cause. Fixed finite-width mean/max pooling does not
inherit this paper's universal-approximation claim. No attention model was
added, no frozen comparison changed, and no theorem supplies missing PDMS,
sample-efficiency or deployment-safety evidence for our models.

## Pseudo-Simulation: two-stage protocol, not continuous replanning

Source: [Pseudo-Simulation for Autonomous Driving, arXiv v3](https://arxiv.org/abs/2506.04218v3),
published in [CoRL 2025 / PMLR 305](https://proceedings.mlr.press/v305/cao25a.html).
Read all 15 PDF pages (main text pp. 1-10, references pp. 11-15) and visually
inspected all rendered pages, including Figures 1-4, Tables 1-3 and Equations
1-2. This is not a read of the separately linked supplement: the publisher's
OpenReview link returned a browser-verification challenge. The local primary
PDF hash and review evidence are retained in
`run_attempts/2026-10-09_pseudosim_main_review.json`. Poppler reported a missing
Symbol display font; the inspected equations/tables remained readable. No
baseline or benchmark-v2 feasibility experiment was executed in this review.

The protocol queries twice for four-second trajectories, not continuous
replanning. Vehicles react through IDM; other actors replay recorded motion.
Stage 2 uses pre-generated synthetic poses and plausible histories after
unrolling four seconds. Gaussian weighting uses the simulated Stage-1 endpoint;
the combined score multiplies Stage 1 by the weighted Stage-2 mean. EPDMS has
nine subscores with human-violation filtering. The nuPlan correlation experiment
omits TLC/LK/EC and uses 244 initial / 4,164 synthetic observations, distinct
from navhard's 450 / 5,462. Table 2 is a March-2026 snapshot, not a current
leaderboard. The paper does not establish real-world deployment correlation.

Our inference for this project: structured-input compatibility must be tested
on synthetic ego histories and coordinate frames, not inferred from camera
rendering quality or obtained by replacing the old scorer. One-stage navtest
results must not be labelled official two-stage NAVSIM v2 results. The concrete
prospective checks are in `CROSS_PROTOCOL_FEASIBILITY.md`; they do not change
the frozen core-study thresholds, inputs or models. The reported planner-count
detail remains unresolved: Figure 3 says 37 rule planners, while the listed
10+15+15 variants sum to 40; do not invent an exclusion explanation without
the separate supplement.

## Cross-benchmark correlation: published method means, not our validation

Source: [Do Open-Loop Metrics Predict Closed-Loop Driving?](https://arxiv.org/abs/2605.00066v1),
an arXiv preprint. Read all ten primary PDF pages: main text pp. 1-8,
references beginning on p. 8 through p. 10; visually inspected every page,
including Figures 1-3, Tables 1-5 and Equations 1-5. No appendix appears in
this PDF, and the underlying papers/checkpoint configurations were not
independently audited or rerun. Poppler's Symbol-font warning is retained;
the inspected formulas and tables are readable. The primary PDF/page hashes
and review limits are in `run_attempts/2026-10-09_crossbenchmark_review.json`.

The study compiles fifteen published methods but analyzes eight complete
PDMS/DS pairs, not eight complete EPDMS pairs. Table 1's seven L2 methods use
CARLA-v2 scoring; Tables 2-4 use the distinct Bench2Drive protocol. Section 7
discloses configuration/architecture confounding, small samples, mostly
single-run closed-loop scores and no causal analysis. Equation 5's simplified
proxy matches PDMS ranking on that same eight-method sample, not a new holdout.
TTC/comfort saturation is qualified to those methods. Progress correlations do
not establish that increasing EP causes better closed-loop driving.

Our read-only arithmetic check of Tables 3-4 reproduces the rounded reported
rank correlations and the three PDMS/proxy inversions. Its exact two-sided
eight-rank permutation p is 0.00456349 (40,320 permutations), distinct from the
reported 0.002; no author implementation or p-calculation method is inferred.
This verifies only published table arithmetic, not underlying experiments or
our models. Values and the calculation boundary are retained in
`run_attempts/2026-10-09_crossbenchmark_table_check.json`.

Our inference for this project: retain every registered official submetric and
the per-scene official PDMS mean; do not substitute a product of average
submetrics, tune proxy weights or alter the registered gates after outcomes.
An NC/EP trade-off is descriptive, not proof of a real slowdown/deadlock
mechanism. Some source shorthand must not become our protocol definition:
TTC is weighted, not a multiplicative gate, in its own Equation 1; benchmark-v2
traffic/stage semantics must follow the pinned official code and Pseudo-Simulation
review, not a blanket non-reactive/four-second description. The discussion's
slow-penalty explanation is not episode-level causal evidence for our models.

## Traffic Element Awareness: regulatory semantics, not eight tracked objects

Source: [Plug-and-Play Traffic Element Awareness, arXiv v1](https://arxiv.org/abs/2608.18035v1).
Read main pp. 1-15, selected appendix pp. 26-39 and 45-50, plus the
opening of Appendix F on p. 40. Visually inspected the 35 full selected pages,
including Tables 1-7 and A.1-A.6. This is not a full 50-page review:
references, extended related work and the additional qualitative gallery are
outside the selected scope. The PDF has recoverable dictionary/string and font
warnings; its original bytes are preserved, not repaired. Evidence is in
`run_attempts/2026-10-09_traffic_elements_review.json`. Author-reported ECCV
acceptance is not an independent proceedings check or a reproduced baseline.

Traffic elements here are regulatory lights/signs, excluding vehicles/pedestrians.
NAVSIM pseudo-labels use an OpenLane-V2-trained detector and depth/geometry;
the planner uses predicted features, not annotation states at inference.
NAVSIM adds TE only, without topology prediction; DriveTransformer instead
retains existing TE modeling and adds topology conditioning. These are not
identical input interventions. Table 3 reports navhard-two-stage; dagger rows
precede an official metric fix. TE-only increments are LTF 25.1 to 28.9,
DiffusionDrive 29.4 to 32.7 and DrivoR 48.3 to 51.8. The roughly ten-point
description must not replace these 3.8/3.3/3.5-point contrasts: combined
TE+SimScale versus plain baselines also changes training data. Table A.1's
100-frame/20-scene/1,126-instance pseudo-label check is bounded, not exhaustive.
Appendix H discloses upstream quality dependence, unmodeled state transitions
and limited long-tail closed-loop coverage.

Our inference: current tracked-object effects cannot establish regulatory-cue
redundancy or general perception redundancy. The privileged eight-object branch
is neither this paper's sensor-to-TE prediction pipeline nor lane/light topology.
Its gains cannot be ranked against our differently budgeted MLPs or relabeled
our results. Inference-stage corruptions also differ from our retrained donor
controls. Keep training-data, supervision, encoder and evaluation changes
separate when discussing mechanisms. Our v1 PDMS lacks TLC; the prospective
v2 audit must retain all official metrics and traffic/stage semantics. This
review does not justify adding a TE branch, changing frozen gates, claiming
v2 feasibility, or inferring deployment safety. The source's 12,147 navtest
count does not change our audited 12,146-token cohort; overlap needs actual
token evidence, not an invented exclusion explanation.

## Claim boundaries for this project

The separate 2026-10-09 primary metadata check covers all ten papers linked by
the plan, with source/version/venue details in
`reference_metadata_2026-10-09.json` and `references.bib`. All ten selected-text
reviews above retain their individual scope limits; Pseudo-Simulation's separate
supplement was not read; Cross-Benchmark Correlation is not a replication of
its underlying cited models; Traffic Element Awareness has selected main and
appendix review, not a full 50-page read. Set Transformer includes all publisher
main/supplement pages, not author-code or independent proof reproduction.
No reference remains metadata/abstract-only, but this does not remove the
individual reading limits or establish exhaustive novelty. The formal Ego Status CVPR 2024 and Pseudo-Simulation CoRL 2025
entries are confirmed in the [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Is_Ego_Status_All_You_Need_for_Open-Loop_End-to-End_Autonomous_CVPR_2024_paper.html)
and [PMLR](https://proceedings.mlr.press/v305/cao25a.html) proceedings. PlanTF's
[ICRA 2024 publisher title](https://ieeexplore.ieee.org/document/10611364/) uses
"Planners"; its selected review remains the singular-title arXiv v1. Neither
metadata checking nor earlier selected-text reading constitutes a final citation
audit of the still-unwritten manuscript.

| Proposed claim | Evidence required here | Status |
|---|---|---|
| Current objects hurt this concatenation model | M1-M0, matched budget and paired seeds | Awaiting complete core scoring |
| Set encoding changes the object effect | M4-M1, convergence and parameter disclosures | Awaiting complete core scoring |
| The improvement uses scene-aligned information | M4-M5 and M1-M2; same-city auxiliary controls and exactly matched eligible coverage for a key alignment claim | Awaiting complete core scoring; base donors can also change city |
| Data budget changes the effect | Nested-session curves with fixed updates and epoch sensitivity | Gated follow-up |
| Effects differ across traffic situations | Frozen input-only strata, counts, metrics and uncertainty | Awaiting scoring |
| The same pattern holds under a new protocol | Separate v2 environment/caches and input-coordinate audit | Conditional extension |

If the controls do not separate these explanations, deliver the planned
reproducible technical report and state the inconclusive result. Do not retrofit
the manuscript around a favorable isolated seed or present Deep Sets as a new
architecture. This restriction preserves the original plan's stopping rule.
