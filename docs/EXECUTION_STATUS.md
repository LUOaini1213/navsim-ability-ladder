# Paper-plan execution record

This is a living execution record, not a completion or publication claim.
The original scope remains [the paper plan](PAPER_PLAN_2026-10-09.md).

## Verified deliverables

- P0 canonical agent: optional feature blocks; ego-only pose execution bypasses
  map/box work; explicit last-call and aggregate diagnostics; optional strict
  failure mode; endpoint-clipping indicator. Historical devkit copies, scores
  and checkpoints preserved.
- Full feature-archive audit: 103,288 navtrain / 12,146 navtest scenes, zero
  token/log/session overlap, finite inputs and targets, binary masks and zero
  raw padding. 27 archived models match training-only normalization exactly and
  produce finite cached-navtest outputs.
- Real-scene audit: 48 deterministic samples in 24 logs; features/targets and
  four live seed-0 agent outputs agree with cached computation. The fixture tests
  verify known rotations against upstream geometry, map independence and failure
  diagnostics. Sampling does not establish exhaustive historical fallback rates.
- Velocity convention: absolute object velocity expressed in the ego basis, not
  ego-relative velocity. 3,285 real track-pair finite-difference diagnostics have
  median error 0.131 m/s, p95 0.553 m/s; these are diagnostics, not exact identities.
- Official score-path parity: 128 agent/scene comparisons, maximum submetric
  error 4.96e-7.
- Both log- and original-session-cluster statistical reports. The no-map box
  effect is uncertain under session clustering.
- Frozen v2 protocol and deterministic cross-session donors. Train/val/confirm
  have 130/16/16 original sessions and 83,099/11,074/9,115 scenes. Thresholds and
  three primary contrasts were fixed before v2 outcome comparisons.
- Six model conditions implemented. All parameter counts are within 10%; masked
  Deep Sets handles empty sets and ignores padding. Current-input matching never
  uses future targets and never crosses partitions.
- Fixed-update training has atomic checkpoints, optimizer/sampler/RNG recovery,
  per-component validation losses and inference/order-sensitivity diagnostics.
  Smoke artifacts are explicitly ineligible for paper evaluation.
- Strict prediction export gates confirmation on all 18 verified core models.
  Scoring uses the original NAVSIM function/config, checks complete aligned token
  coverage, records failures and supports per-token SQLite recovery.
- Checkpoint recovery unit test restores model, Adam moments, sampler position
  and RNG and exactly reproduces the uninterrupted next update. The confirmation
  gate rejects an incomplete core manifest. All 7 agent-audit and 9 mechanism
  tests pass in the NAVSIM environment; repository code compiles successfully.
- A separately labelled 3-update Deep Sets smoke model was exported on navtest
  and scored on three tokens without exception. Its artifacts remain explicitly
  ineligible for paper results; no score from that smoke is a scientific finding.
- The complete 43-test suite passes in the NAVSIM environment with no skips.
- Closest-prior review now covers all 23 pages of Driving on Memory, including
  visual checks of its method and key result tables, and PlanTF's main text and
  supplementary result sections. The [comparison record](CLOSEST_PRIOR_REVIEW.md)
  narrows acceptable novelty claims without changing the experiment/stopping plan.
- The primary HTML main text and appendices A-E of Is Ego Status All You Need?
  have also been checked. Its metric/subgroup/input-corruption precedents further
  restrict what can count as a contribution.
- Twenty new analysis/case/gate/artifact-manifest tests pass. Case selection uses
  frozen positive/negative/neutral thresholds, token hashes and one case per
  original session per contrast; missing bins are reported, never filled by
  relaxing thresholds. All three seeds are displayed individually.
- The complete 63-test suite passes in the NAVSIM environment with no skips.
  Four additional donor/OS-lock tests separately pass, including cross-process
  exclusion and waiting until the current owner releases its lock. Historical
  report tests emit pre-existing unclosed-file warnings but no test failures.
- After the new input/runner checks and provenance tightening, the complete
  67-test suite also passes in the NAVSIM environment with no skips (214.9 s).
  Prediction receipt verification confirms unchanged complete validation coverage.
- Numeric follow-up gates are now separately frozen. They were added after core
  training began but before full controlled score comparisons; this timing is
  explicit. Negative effects can trigger data-budget curves, not only improvements.
- Input-only donor diagnostics pass partition/session independence. Same-city
  donor rates are 42.4% train, 43.7% validation, 37.1% confirmation and 29.6% navtest.
  The controls can change city distribution as well as scene alignment. They
  cannot isolate dynamic interaction use; a key alignment claim requires a
  separately frozen same-city control or must be withheld. The complete raw
  cohort audit now passes: all 103,288/12,146 selected training/test tokens have
  feature records, with zero missing logs or selected scenes without features.
  This coverage audit does not reconstruct historical fallback reasons.
- The full M0 seed-0 checkpoint exports all 11,074 validation predictions with
  finite outputs and complete provenance. Repeating the export verifies and
  reuses its hash-bound receipt without changing the prediction archive.
  This is inference/export validation, not a PDMS score or completed core result.
  Confirmation export remains gated on every core model.
- The same-city donor generator and separate extension design are implemented.
  Four tests pass for strict city/session constraints, deterministic matching,
  explicit unsupported-city exclusion and current-input-only access. Actual
  auxiliary input freezing/training/evaluation still requires its memory/gate checks.
- The standalone core-figure renderer is implemented with production provenance
  gates and clearly ineligible synthetic layout smoke mode. Scientific rendering
  and visual inspection have not yet happened; source existence is not a figure result.
- Upstream enum inspection confirms generic_object is not an AGENT_TYPE. The
  unchanged selected-object branch contains vehicle/pedestrian/bicycle/generic
  categories; bicycle/generic objects share zero type flags. Methods and scope
  statements now disclose this instead of implying exclusively moving actors.
- A [full-scope requirements ledger](COMPLETION_REQUIREMENTS.md) and
  [methods note](CONTROLLED_METHODS.md) separate verified design from pending
  experiments, rendering and the eventual evidence-grounded manuscript.
- All 18 full-budget core models have completed and pass the actual
  `results/audit_v2/core_manifest.json` check: 13,000 updates each, one shared
  full-data normalizer, and parameter max/min 1.0773. The verified
  `training_diagnostics/summary.json` and `.md` retain per-seed losses, timing,
  order sensitivity and history/checkpoint hashes. These are not PDMS findings.
- Actual process interruption/recovery passes at 1,000 smoke updates after
  interruption at 325: exact model/Adam/RNG/sampler/validation history equality.
  New-model live/cached parity passes on 16 deterministic scenes x four real-input
  seed-0 conditions (64 comparisons, max pose error 7.63e-6). The scope is sampled,
  not every scene/seed or a live physical interpretation of donor controls.
- The new two-token validation cache smoke passes. Full validation caching was
  interrupted for critical memory pressure; confirmation caching and complete controlled scoring are not complete.
  No new PDMS comparison exists at this record update.
- The registered 24-job smaller-data training runner is implemented. Its first
  attempt (session 20056) exited without training when the required validation
  analysis report was absent after the evaluator stopped. It verifies the gate against actual hash-bound validation
  score CSVs, uses 12 fixed-update and 12 equal-epoch jobs only if triggered, and
  does not produce driving-score curves by completing training alone.
- Same-city design/source hashes were bound before any complete core comparison
  report existed (`city_protocol/design.json`, 2026-10-09T06:57:46Z). Actual donor
  generation has not run: its 2,200-MB free-memory guard correctly refused a
  low-memory attempt. Do not change the bound generator silently.
- The renderer's synthetic layout smoke ran and was inspected. Footer overlap
  and empty-row clipping were found and fixed in source; rerender/inspection of
  the corrected outputs and real scientific figures remain pending safe memory.
- Six selected primary papers are covered by the closest-prior record, including
  PDM/NAVSIM official supplements and selected Deep Sets theory/architecture
  appendices. The six-entry bibliography contains no invented publication claim.
  PDM centerline baselines and NAVSIM input/data ablations further constrain novelty.
- The new city-design/curve-gate/training-summary/plot-fixture tests pass separately;
  the last full suite was 67 tests before these 17 additions. An updated complete
  suite and final scientific/reproducibility audit have not yet run.
- A current [reproducibility guide](REPRODUCIBILITY.md) and runtime snapshot record
  the real CPU environment, 201 installed dependency versions, 160 tracked
  upstream hashes and the preserved local dataloader patch. The snapshot is
  current evidence, not a retrospective pre-launch source immutability claim.
- The critical-memory attempt is now preserved separately. Nine resource-guard
  tests pass, including actual owned-child termination while an unrelated child
  remains live. The complete 93-test suite passed in 174.841 seconds with no
  skips before the subsequent curve/city/cache-recovery test additions.
- Corrected synthetic layouts were rerendered into a separate folder and all
  three PNGs visually inspected; they remain scientifically ineligible. The
  real 18-model convergence PNG was also inspected with its receipt/history/model
  hashes bound in `figures/training/visual_qa.json`. Independent SVG rasterization
  remains uninspected; no real PDMS figure exists yet.
- Actual same-city donors are now frozen and independently audited in
  `results/audit_v2/city_inputs.json`. Training stays 83,099 scenes/130 sessions;
  validation is 10,312 scenes/15 sessions (762 Singapore scenes excluded),
  confirmation 8,379/15 (736 excluded), and navtest 12,146/44 without exclusions.
  All eligible mappings have same-city and different-session fractions of 1.0.
  The inherited base diagnostic limitation string is explicitly qualified by the
  separate audit; frozen sources/receipts were not rewritten. No auxiliary model
  has been trained or scored.
- Actual isolated cache recovery passes three deliberate faults (empty file,
  truncated LZMA and invalid pickle) in a newly created two-scene test directory.
  Each restored file is byte-identical to its baseline, has identical checked
  semantics and zero official-submetric error for the fixed anchor trajectory;
  the untouched second file remains unchanged. Formal/historical cache roots were
  not targets of mutation. This is a recovery audit, not full cache coverage or a
  scientific performance finding (`results/audit_v2/cache_recovery.json`).
- Guarded curve evaluation and paired analysis are implemented separately from
  the base core analyzer. They require all 24 follow-ups plus six reused anchors,
  aligned score coverage, subset normalizers and the verified actual-score gate
  before auxiliary confirmation. Fixed-update/equal-epoch curves share the same
  full-data anchor; four small-minus-full encoder-effect sensitivity contrasts
  are paired by scene/session. Execution and real curve figures remain pending.
- The [benchmark-v2 feasibility checklist](CROSS_PROTOCOL_FEASIBILITY.md) is
  grounded in the official version/metric/install documents. It separates v1 PDMS,
  v2 one-stage EPDMS and two-stage aggregation and requires exact commit/fix,
  data-overlap, synthetic-state and measured-resource checks. None has run yet.
- The complete 105-test suite passed in 143.521 seconds, with a retained command,
  guard and output log under `run_attempts/2026-10-09_full_suite_105.*`. Two upstream
  baseline-binding and two cache-resumption tests subsequently passed separately;
  the updated 109-test suite has not run as one complete invocation. Repository
  sources compile and `git diff --check` passes (line-ending warnings only).
- All 160 tracked devkit files still match the preserved current-runtime snapshot.
  The new controllers bind upstream files/configuration, including the legitimate
  existing dataloader patch, and refuse changes before later stages. This is a
  current-baseline check, not proof of historical/pre-training source immutability.

## Active execution

The 18-model core training runner has been launched with two CPU threads per
child, serial jobs, below-normal child priority and a free-memory guard.
Authoritative progress is the live process/tool session plus checkpoint hashes;
`results/mechanism_v2/core_status.json` alone is only a progress record, not proof
that a process is alive. Logs/checkpoints are under
`E:/navsim_workspace/exp/mechanism_v2`.

Core training is finished (18/18); the runner exited successfully and all
artifacts were independently verified. This is a training milestone, not a new
PDMS claim. Later evaluation state is available in the evaluator's log/manifest.

The initial serial evaluation attempt (session 71587, PID 26060) exited with
code 1 after deliberate termination of its owned cache child at critical RAM
pressure (392/284/346 MB free). No unrelated job was stopped. The curve runner
then exited through its missing-validation-report dependency gate, with zero
training jobs. Their terminal manifests are preserved under
`results/mechanism_v2/run_attempts/2026-10-09_memory_stop/`, together with an event
ledger. At the later inventory there were 1,489 partial new-cache files (592.1 MB);
file presence does not prove valid complete coverage. No material file was deleted.

The evaluation supervisor now requires 2,400 MB continuously for 20 seconds before
launch and monitors the owned child every five seconds. It stops that child tree
at one observation below 512 MB or two consecutive observations below 1,000 MB.
Only the resumable full validation/confirmation cache stages can automatically
wait and retry after a recorded memory interruption; other stages or ordinary
nonzero exits require explicit review. Every retry retains its attempt/failure
record and validates existing cache files. Scientific model/scoring sources and
the frozen protocol remain unchanged. Testing and a new live launch must be
verified separately; a status JSON alone is never proof of liveness.

Artifact/input audits, actual process recovery, live parity and cache smoke have
passed. Full scoring remains pending. A completed evaluator would cover core
evaluation only, not the complete paper-plan goal.

The guarded retry (session 44637, runner PID 40308) has also exited with code 1:
repeating `input_controls` reached 500.33 MB free and the watchdog stopped only
the owned child tree. The terminal attempt is preserved under
`run_attempts/2026-10-09_watchdog_stop/`. The second queued curve attempt (session
14889, PID 40316) then correctly exited through the absent-validation-report
dependency gate, again with zero training jobs. Neither process remains live.

The earlier completed full input/recovery/live/cache-smoke audits remain present
and pass; their source/protocol hashes have not changed. Cache work is now isolated
in `resume_mechanism_cache.py`, which checks that original successful attempt,
re-verifies all 18 models, binds the actual audit receipts and upstream baseline,
and explicitly does not rerun or retrospectively relabel those completed audits.
It builds only validation/confirmation caches, preserving attempt failures and
retrying only recorded memory interruptions. Scoring remains separate and gated.

At the earlier 2026-10-09T07:50:04Z observation the cache-only controller was live (actual PID 41048,
launcher PID 40736, session 60359), **waiting for memory with no cache child**.
Measured free RAM was 1,642 MB, below its continuous 2,400-MB launch condition.
Its authoritative attempt receipt is
`run_attempts/cache_resume_20261009T074804Z_41048.json`; `cache_resume_status.json`
is only a progress snapshot. No new full PDMS score or active curve training was
claimed. Do not rerun bound audits/modify frozen files while a live controller owns
the heavy-job lock; review source/receipt changes explicitly instead.

The donor city/session audit also identifies an important extension constraint:
Singapore has only one original session in each validation/confirmation partition.
A strict same-city cross-session control cannot evaluate those partitions there;
that coverage loss must be explicit. All training cities have at least 12 sessions,
so a same-city training donor pool is feasible without removing a training city.

## Later memory-safety stop and conditional city implementation

The cache controller actually started validation. Attempt 1 stopped at two
observations of 838.73/900.21 MB. Attempt 2 launched with 3,152.84 MB and reached
5.36 MB free, triggering its emergency stop. Attempt 3 subsequently launched;
because this automatic retry pattern risks another system freeze, its verified
owned launcher tree (40736, actual runner 41048) and the queued scorer tree
(42412, actual runner 39092) were explicitly stopped. Both tool sessions ended
with exit 1. No cache/model file or unrelated process was removed. The unfinished
third cache child's terminal code was not independently measured.

Terminal external-stop annotations and the two raw pre-stop snapshots are retained
under `run_attempts/2026-10-09_cache_headroom_stop/`; the old cache supervisor source
is preserved there with its original SHA256
`3d67cd2287fdd78158961f625bec7bb9810091a97ddee32e299b3a2c71ad35b6`.
The progress snapshots alone are not evidence these terminated runners remain live.

The cache-only supervisor now requires 6,000 MB continuously for 30 seconds,
checks its owned child every 0.5 seconds, stops it at an observation below
2,000 MB (emergency below 1,500 MB), and **does not automatically retry a memory
interruption**. This is an operational policy change after actual pressure, not
a scientific protocol/model/scorer change. It cannot guarantee immunity to
unrelated allocations. Its dedicated policy/error tests passed; any new launch
must use a new attempt receipt and independently verify liveness.

The independent scoring controller's six fixture tests passed. It requires both
full cache receipts, reuses source-bound completed audits without repeating the
high-memory raw input audit, and schedules exactly 54 core exports. Its first
queued attempt started zero child scoring jobs before the external stop.

The same-city execution rule was frozen at 2026-10-09T08:39:33Z before complete
core comparison reports: absolute validation M1-M2 or M4-M5 >=0.01 triggers all
six auxiliary models. Both signs may trigger; no trigger withholds the clean
alignment interpretation. This extends the execution policy rather than
pretending the numeric rule was in the original text. Actual decisions recompute
the effects from all 18 core CSVs and check source/model/protocol hashes.

Separate city training/export/evaluation/analysis sources are implemented. The
confirmation entry requires all 18 base and six auxiliary full-budget artifacts;
the 12 fitted cells use identical eligible tokens/logs, and real-anchor predictions
must equal the corresponding full-archive subset. Three session-paired auxiliary
contrasts include the alignment-by-encoder difference of differences, conditional
on three fitted seeds. Auxiliary-family intervals do not establish unconditional
global confirmatory coverage. The 25 newly added city fixture tests separately
passed; the combined city-pattern suite then passed all 34 tests in 2.000 seconds,
with its command, guarded exit-0 receipt and complete log retained at
`run_attempts/2026-10-09_city_fixture_suite.{json,log}`. Scientific city models/scores, figure QA and the updated full suite remain
pending. Actual read-only checks still verified 18 core models, city validation
10,312 scenes, and the original 15 source/policy bindings; bound audit files were
not overwritten.

After the safety-policy tests, a new cache-only attempt was launched:
`run_attempts/cache_resume_20261009T085623Z_30252.json`, session 94130,
actual runner PID 30252 (launcher 40608). The new scorer is queued in
`run_attempts/core_scoring_20261009T085658Z_37152.json`, session 27613,
actual PID 37152 (launcher 36956). Both were independently found in the OS
process inventory. The observed cache state is **waiting_for_memory, no child**;
the scorer is waiting for its lock with no child. The initial free-RAM reading
was 3,720.75 MB, below the new 6,000-MB launch condition. These live waiting
controllers are not complete caching/scoring evidence. The cache supervisor's
four tests, including the effective first-low-sample 2,000-MB stop, passed;
compileall and git diff --check passed. The updated full suite still has not run.

## Follow-up rendering and separate component checks

The actual full-budget training SVG was independently rasterized with
sharp/libvips/librsvg and its entire raster was visually inspected: four panels,
metre/radian units, legends and min/max-seed scope are readable. The unchanged
scientific SVG/receipt and the independently generated raster are hash-bound in
`figures/training/vector_visual_qa.json`. This closes vector layout QA for training
diagnostics only; it supplies no PDMS or deployment-safety evidence.

The city/curve analyzers now expose reusable source-rebuilding functions. City
analysis requires its actual execution gate before loading features; curve
analysis checks the current scoring configuration and an explicit workspace.
`plot_followups.py` rebuilds the report from model/score/prediction provenance
before rendering, rather than trusting an edited summary JSON. It adds seven
city/curve contrast, stratum, submetric and budget figures. Both pending evaluation
runners include this rendering stage, but rendering is not visual acceptance.

Separate observed tests passed: plot helpers 8, builder gate-order fixtures 2,
curve analysis 7, city analysis 5, and the installed SVG rasterizer's Node tests 4
with zero skips. Their scoped summary is
`run_attempts/2026-10-09_followup_component_tests.json`; stdout is available in the
tool outputs, not an independently retained complete log. These are not a rerun
of the full suite or evidence that auxiliary experiments have completed.

All seven first-pass synthetic layout PNGs were inspected and were readable.
Review found that the fixture did not exercise wider multiplicity-adjusted
intervals, and small negative heatmap values displayed as "-0.000". The original
renderer is preserved as `run_attempts/2026-10-09_first_followup_layout_renderer.py`.
The fixture interval width and rounded label presentation were corrected without
changing scientific interval calculations; two additional tests pass. Corrected
layout generation completed in session 53169 (exit 0) under the resource guard,
in the two `tmp/followup_layout_smoke_*_20261009_v2` directories. All seven corrected
PNG outputs were then inspected, including a separate original-detail view of
the encoder curve. Wider adjusted intervals and rounded zero labels are visibly
correct; titles/captions, empty/single-session strata and reused full-data anchors
are readable. The source, render receipts and inspected PNG hashes are bound in
`run_attempts/2026-10-09_followup_layout_png_qa.json`. All fixtures remain
scientifically ineligible. Independent fixture SVG rasterization then completed
in guarded session 67570 (exit 0). All seven independent SVG rasters were actually
viewed, including the complete encoder curve separately, and their source,
fixture-receipt, independent-receipt and raster hashes rechecked. Titles, captions,
ordinary/adjusted intervals and heatmap labels are readable. This separate
acceptance is bound in `run_attempts/2026-10-09_followup_layout_vector_qa.json`;
it closes synthetic fixture PNG/SVG layout QA, not scientific figure requirements.
The eight plot helper tests were rerun and passed in 0.004 seconds, and the seven
corrected PNG hash bindings also passed an actual post-inspection audit.

The user elected to wait for other tasks to release RAM. The cache/scoring
controllers remain live waiting, with no child reported; no unrelated task was
stopped and the 6,000-MB-for-30-seconds cache launch condition is unchanged.
Brief RAM windows allowed small component/fixture work, not full caching. This
waiting state is not evidence of experimental completion.

## Local source snapshot component

The README now separates the archived-v1 report test from the complete newer
suite, which may import the ML runtime/use controlled-study inputs. No claim that
archived report reproduction also reproduces the new experiments is retained.

`analysis/repro_source_bundle.py` creates a fresh, local source/instruction ZIP
with deterministic member ordering/timestamps and per-file sizes/SHA-256 values.
It preserves existing/failed output attempts, restricts output to a new child of
`results/repro_snapshots`, excludes datasets/caches/weights/paper PDFs/private
environment files, and streams verification without extracting archive paths.
Its source-only completion flags cannot be promoted to an experimental release.
Nine component tests passed in 0.339 seconds; retained stdout, source hashes and
the initial newline-fixture correction are recorded in
`run_attempts/2026-10-09_source_bundle_component_receipt.json`.

An actual 127-file, 752,794-byte source snapshot was built at
`results/repro_snapshots/source_components_20261009`, then independently checked
with the documented `verify` command. Its manifest/ZIP hashes are recorded in
the component receipt. All 98 Python members were additionally compiled directly
from the verified ZIP with Python 3.9.25, without executing their imports; the
separate syntax-only receipt is `run_attempts/2026-10-09_source_bundle_syntax.json`.
This is a snapshot of bytes at capture, not a promise that
subsequent documentation edits leave the whole worktree unchanged. No upload,
dataset redistribution, formal experimental smoke, full-suite pass or final
package/project completion is claimed. Complete scores, applicable extensions,
artifact terms review, manuscript and the final reproduction audit remain.

## Current-plan bibliography metadata check

All ten paper IDs in the plan were rechecked against primary arXiv, publisher or
author pages. Four previously absent bibliography entries were added: Pseudo-
Simulation, Set Transformer, Cross-Benchmark Correlation and Traffic Element
Awareness. The CVF record confirms Ego Status's CVPR 2024 venue/pages; PMLR
confirms Pseudo-Simulation's CoRL 2025/PMLR 305 entry. PlanTF's ICRA 2024 publisher
title is plural "Planners", while the selected arXiv v1 review retains its singular
title/year/version URL. The arXiv v3 revision of Pseudo-Simulation is not a NAVSIM
benchmark-version identifier. Traffic Element Awareness is recorded as an arXiv
reference with its authors' ECCV acceptance statement, not invented proceedings.

`docs/reference_metadata_2026-10-09.json` records the sources and review boundary:
six selected-text reviews with their existing documented limits, four metadata/
abstract-only sources. `run_attempts/2026-10-09_reference_consistency.json` records
an actual standard-library consistency check of all ten title/ID/year/type/URL/
author-count entries and full plan-ID coverage, with hashes of the current files.
This check is controlled-format, not a generic BibTeX parser or automatic author-
identity audit. The earlier "abstract only" Memory row was updated to reflect its
already completed selected full-text review. No final manuscript citation audit,
exhaustive novelty review, reproduced literature baseline or scientific outcome
is claimed; the final manuscript is still unwritten.

## Methods-note reconciliation with actual artifacts

`CONTROLLED_METHODS.md` no longer calls the completed selected-raw-cohort audit
pending. It reports its actual 103,288/12,146 selected-scene coverage and
1,192/136 available-log counts, while retaining the distinction from all raw
frames and unknown historical extraction/fallback reasons. The previous raw/city
audit receipts were reused read-only, not overwritten or newly represented as
pre-training immutable evidence.

The note now distinguishes the verified 18 full-budget, 13,000-update core
models and 1.0772576 parameter ratio from still-missing PDMS evidence. It adds the
strict same-city common cohorts (83,099 train; 10,312 validation; 8,379 confirmation;
12,146 navtest), 762/736 Singapore exclusions and unchanged-training anchor reuse.
It also states the actual learning-curve/equal-epoch, seed-sensitive, city and
cross-protocol resource/claim rules and their non-retrospective/exploratory scope.

An actual lightweight read-only check reran the full 18-model artifact verifier,
rechecked city input lineage and raw-audit/protocol bindings, and confirmed no
complete core score report exists. The updated note and evidence hashes are in
`run_attempts/2026-10-09_method_note_refresh.json`. This is not a final manuscript,
fresh raw-data audit, official score result or complete project delivery.

## Prospective five-seed registration component

`analysis/seed_followup.py` now binds a separate extension design to the unchanged
base protocol, donors, original follow-up policy and training sources. Its actual
registration at `seed_protocol/registration.json` was captured at
2026-10-09 10:38:39 UTC, before complete controlled score reports existed. This is
not a claim that the original prose plan already specified every execution detail.

If a registered primary contrast has paired seed effects at or below -0.01 and at
or above +0.01 in validation or confirmation, the design adds both seeds 3/4 for
all six conditions: twelve full-data, 13,000-update jobs, with the original three
seeds retained. No seed, favorable contrast or checkpoint may be selected after
seeing outcomes. The in-memory protocol recipe preserves every base cohort,
donor policy and training setting except the explicit seed extension; no new
training protocol/donor copy or model was materialized by this component.

Activation verification requires both complete validation and confirmation score
archives, the exact eighteen fitted model identities, matching provenance and the
actual hash-bound `gate_confirmation.json`. It recomputes the primary paired seed
effects from all eighteen CSVs per split, rejects smoke/missing/changed/invalid
evidence, and does not turn a passing gate into an added-model completion claim.

Fifteen scoped tests actually passed without importing NumPy or Torch; their full
stdout is retained at `run_attempts/2026-10-09_seed_registration_tests.log`. The
actual component receipt is `run_attempts/2026-10-09_seed_registration_component.json`.
Fourteen cache-bound analysis source hashes were rechecked unchanged. This
registration-only component did not implement training or evaluation. The
subsequent training-entry work is recorded below; actual extension artifacts,
normalizer parity, score analysis and final deliverables remain pending.

## Guarded five-seed training entry

`seed_models.py` now verifies the actual core gate before materializing a separate
`seed_protocol/active/protocol.json` and byte-identical donor copy. The original
core inputs are untouched. Existing valid copies are checked, not rewritten;
partial or changed copies are preserved and rejected. No actual local activation
exists yet because complete core score evidence is still absent.

`run_seed_training.py` holds the existing heavy-job OS lock and sends the fixed
twelve jobs to the unchanged trainer under an isolated `seed_models` output
root. Launch requires 6,000 MB continuously for 30 seconds; the owned child is
checked every 0.5 seconds and stopped below 2,000 MB (1,500 MB emergency), with
zero automatic retries. Ordinary failures are not skipped; a later explicitly
reviewed invocation can resume only matching existing trainer checkpoints.

The audit rejects smoke/wrong-budget/source/parameter/checkpoint/normalizer
changes. All twelve added artifacts and normalizer parity with the verified core
are required before an eventual evaluator may access added confirmation
predictions. The registration's outcome-gated/exploratory boundary remains.

Thirty lightweight seed-component tests actually passed in 15.788 seconds,
including tiny owned standard-library subprocesses for logging/failure cleanup,
without importing NumPy or Torch. Full output and source hashes are retained at
`run_attempts/2026-10-09_seed_training_tests.log` and
`run_attempts/2026-10-09_seed_training_component.json`. The actual scientific
training runner was not launched; actual new models remain zero. This training
component did not implement prediction/scoring/analysis; their later component
progress is recorded below. The updated full test suite is still pending.

## Five-seed prediction, official-score entry and source-rebuilt analysis

`predict_seed.py` verifies all eighteen original plus twelve added full-budget
models before loading features or importing Torch, including before any added
confirmation inference. It preserves the base/extended protocol identities,
uses the unchanged frozen model/feature transformation, records strict NPZ
receipts and refuses partial/changed/nonfinite/wrong-shape archives without a
silent fallback or overwrite.

The existing official scorer requires a single training protocol per batch.
`run_seed_evaluation.py` therefore sends only the twelve added models to three
separate official batches (validation, confirmation, exploratory navtest), after
36 registered added exports. Original eighteen-run scores and prediction bytes
are reused without relabelling or rescore. This evaluator shares the same heavy
lock and conservative 6,000-MB/no-automatic-retry pipeline as seed training.

`analyze_seed.py` separately verifies both score archives, NPZ receipts, all
thirty model/protocol identities, exact frozen token/log cohorts, all seven
finite submetrics, wrapper/scoring-config/cache identity and the common
training-only normalizer. It then applies the existing paired session/log
bootstrap to all five seeds with fixed 10,000 draws, retaining primary adjusted
intervals, overlapping input strata and original-three versus added-two
diagnostics. Reports explicitly label outcome gating, fitted-seed-conditional
uncertainty and lack of fresh independent confirmation; they do not change the
original three-seed gate or establish novelty/deployment safety.

Fifty scoped seed-component tests actually passed in 17.280 seconds with zero
failures/errors/skips. They include real small NPZ/receipt/seven-metric CSV
fixtures, all-five-seed statistics and a synthetic thirty-run source-rebuilt
report that mocks only the separately tested model-artifact layer. NumPy was
imported for array/statistics tests, Torch was not. The full log/receipt are
`run_attempts/2026-10-09_seed_evaluation_tests.log` and
`run_attempts/2026-10-09_seed_evaluation_component.json`; registered/training
components and fourteen cache-bound analysis sources remain unchanged.

No actual extension protocol was activated, no added model was fitted and no
scientific prediction, official-engine score or five-seed report was generated.
The evaluator was not launched. Scientific five-seed figures/visual QA, actual
execution and the updated full suite remain pending; fixtures do not constitute
an end-to-end experimental reproduction.

## Bounded Pseudo-Simulation primary-paper review

The arXiv v3 primary PDF was downloaded read-only, verified as 15 pages and
5,341,242 bytes, and read in full (main pp. 1-10, references pp. 11-15).
All fifteen rendered pages were visually inspected, including the four figures,
three tables and both equations. The original Poppler Symbol-font warning is
disclosed rather than called a warning-free render. An initial text extraction
failed on console encoding; `-X utf8` fixed it and all pages were then read.
The separately linked OpenReview supplement returned a browser-verification
challenge and was not read. This review used the PDF reading skill only; no
manuscript PDF was created or edited.

`CLOSEST_PRIOR_REVIEW.md` now distinguishes synthetic-state history/coordinates,
simulated reached-endpoint weighting, reduced correlation-study metrics and
public benchmark coverage. `CROSS_PROTOCOL_FEASIBILITY.md` adds inferred
preflight requirements, not an executed adapter or v2 feasibility result.
No frozen core gate, scientific source, model, cache or historical score changed.
The bibliography's current boundary is seven selected-text reviews with limits
and three metadata/abstract-only records; the earlier six/four consistency
receipt is preserved as a historical snapshot, not silently overwritten.
The new PDF/page/document hashes and actual lightweight preservation check are
recorded in `run_attempts/2026-10-09_pseudosim_main_review.json`.

The same cache/scoring controllers remain responsible for the pending full
experiment; no replacement process, conditional v2 run or large data download
was started. Their conservative memory wait is unchanged. Full core scores,
triggered extensions, scientific figures, manuscript and final package remain
unfinished.

## Five-seed source-rebuilt graphics component

`plot_seed.py` now provides four dedicated figure types per split: all seven
paired contrasts with all five seed effects, descriptive original-three versus
added-two means, overlapping input strata including empty/single-session groups,
and all seven submetric effects. Primary adjusted intervals remain confined to
the original three comparisons. Fitted-seed conditioning, outcome gating,
exploratory strata, DDC's diagnostic-only status and lack of fresh confirmation
or deployment-safety evidence are explicit; subset means do not create new CIs.

Before production plotting, the report must match an actual thirty-run rebuild
through `analyze_seed.build_report`, retaining distinct original/added training
protocols and both official score archives. Production headroom is checked at
6,000 MB before heavy imports or rebuilding. Synthetic layouts require 2,000 MB,
tmp-only destinations and no scientific input, and are stamped ineligible.
Receipt-bound completed bytes can reuse; changed/unrelated/partial outputs are
preserved and refused. A changed report during rendering cannot receive a new
completion receipt. Successful rendering alone is not visual QA.

`run_seed_figures.py` requires the actual completed seed-evaluation attempt and
unchanged bindings, owns the shared heavy-job lock and dispatches the three
splits through unchanged `SeedPipeline` (6,000 MB stable 30 seconds, 0.5-second
polling, owned-child stop below 2,000 MB, zero automatic retries). A fresh
`--out-root` version is required after a partial or changed render.

Twenty-three new tests and eight existing follow-up-plot regression tests pass.
They check report acceptance/rejection and file ownership with mock rebuilds,
mock runner dispatch and placeholder bytes inside temporary directories. These
are not real image-format renders, artifact-level thirty-model reproduction,
official scores or layout QA. Logs and source hashes are retained in
`run_attempts/2026-10-09_seed_graphics_component.json`.

No actual graphics runner or renderer was launched, and no scientific or
synthetic-layout image was generated. The earlier registered/training/evaluation
sources and original cache bindings remain unchanged. Actual extension execution,
rendering, PNG/SVG inspection, updated full suite and final project audit remain.

## Cross-benchmark primary review and bounded table arithmetic

Read all ten pages of arXiv:2605.00066v1 (main text pp. 1-8, references pp. 8-10)
and visually checked every rendered page, including Figures 1-3, Tables 1-5
and Equations 1-5. No appendix appears in the PDF. The Symbol display-font
warning remains disclosed; inspected formulas and tables are readable. This
was read-only PDF review, not manuscript-PDF creation or a benchmark run.

The main sample comprises eight published PDMS/DS pairs, not eight complete
EPDMS pairs; the seven-method L2 table uses a different CARLA-v2 protocol.
`CLOSEST_PRIOR_REVIEW.md` now distinguishes architecture/configuration
confounding, outcome-sample proxy evaluation and lack of causal/model-specific
external validation. The official metric, submetrics and all frozen gates remain
unchanged; no proxy score or inferred slowdown mechanism was added to our results.

A pure standard-library arithmetic check on manually inspected Tables 3-4
reproduced all six rounded rank correlations and the three PDMS/proxy inversions.
Exhausting 40,320 permutations gave exact two-sided PDMS/DS p=0.00456349;
the paper reports 0.002, and its calculation method was not inferred. The
transcribed values and exact calculation scope are preserved separately in
`run_attempts/2026-10-09_crossbenchmark_table_check.json`. This is published-table
arithmetic only, not verification of cited checkpoint/configuration lineage,
rerun benchmarks, our models or a new independent holdout.

Current bibliography scope is eight selected-text reviews with stated limits
and two metadata/abstract-only references. Earlier six/four and seven/three
receipts remain historical snapshots; the new source/page/document bindings
are in `run_attempts/2026-10-09_crossbenchmark_review.json`. Actual core scoring,
triggered extensions, scientific figure QA, manuscript and final package remain
unfinished. The original resource-wait controllers were not replaced.

## Bounded Traffic Element Awareness review

Read the main pp. 1-15 and selected appendix pp. 26-39 and 45-50 of
arXiv:2608.18035v1, plus Appendix F's opening on p. 40. All 35 full selected
pages were visually inspected, including Tables 1-7/A.1-A.6. This is not a
full 50-page read, author-code/checkpoint reproduction or proceedings check.
The original dictionary/string and font warnings are retained and disclosed;
no repaired PDF or manuscript PDF was authored. A source unresolved figure
cross-reference on p. 31 was observed, not silently corrected.

The closest-prior comparison now distinguishes regulatory TE semantics from
the privileged eight tracked objects, NAVSIM's TE-only integration from
topology-enabled settings, TE-only from TE+SimScale published contrasts, and
inference corruption from independently trained donor controls. Its derived
Table 3 subtraction is published-table arithmetic, not an experimental result
of this project. No TE branch, input, training-data policy, score, frozen gate
or scientific source was changed.

Current bibliography scope is nine selected-text reviews and one
metadata/abstract-only reference (Set Transformer). Prior review/count receipts
remain historical snapshots. Source/page/document hashes, preservation checks
and scope exclusions are retained in
`run_attempts/2026-10-09_traffic_elements_review.json`. The existing controllers
continue the user-authorized resource wait without duplicate work. Actual core
scoring, applicable extensions, scientific figure QA, manuscript and final
reproducibility package remain pending.

## Set Transformer publisher main/supplement review

Read all ten publisher main-PDF pages and all seven supplement pages, then
visually inspected every rendered page, including architecture, theoretical
statements, all main/supplement tables and the dummy-set runtime figure.
Both pdfinfo and both Poppler renders completed without parser/font warnings.
An initial image lookup used the wrong zero-padding convention; `rg --files`
resolved the actual names before all seventeen visual checks completed. No
PDF was edited or authored and no author experiment/theorem was reproduced.

The current `MechanismModel` was read-only checked: M4/M5 use a row-wise
MLP and masked mean/max, not SAB/ISAB/PMA. The related-work note distinguishes
equivariance/invariance from empirical interaction reasoning, architectural
PMA seeds from fitted training seeds, and a capacity-dependent constructive
universality argument from finite-model guarantees. It also retains source
main/supplement numerical differences without inventing a checkpoint history.
No attention model, scientific source, score or frozen gate changed.

All ten planned references now have selected-text comparisons; individual
scope exclusions remain, and this is not an exhaustive novelty certificate or
the final citation audit of the still-unwritten manuscript. Earlier six/four,
seven/three, eight/two and nine/one review receipts remain historical snapshots.
PDF/page/document hashes and lightweight preservation evidence are retained
in `run_attempts/2026-10-09_set_transformer_review.json`. Original cache/scoring
controllers continue the user-authorized RAM wait. Complete scoring, applicable
extensions, scientific figure QA, manuscript and final package remain pending.

## Post-seed source-only snapshot refresh

The historical v1 source ZIP predates fifteen seed-component source/test files
and omits the public bibliography metadata referenced by the review guide.
The new v2 recipe retains the source-only allowlist and admits exactly
`docs/reference_metadata_2026-10-09.json`, not arbitrary JSON files. Missing
required metadata fails before output creation. Historical v1 verification
retains its original allowlist and does not rewrite its archive or manifest.

Thirteen component tests passed in 0.502 seconds, including legacy compatibility,
new-schema required metadata, exclusion of unrelated/private JSON, byte integrity
and preservation of existing/failed targets. The test process imported no Torch,
NumPy or Matplotlib. Retained stdout is
`run_attempts/2026-10-09_source_bundle_v2_tests.log`.

An actual new 143-file, 960,612-byte snapshot was built at
`results/repro_snapshots/source_components_post_seed_20261009`, then separately
verified through the CLI. All fifteen seed files matched current source bytes;
the included public metadata has ten references. All 113 Python members compiled
directly from the ZIP with Python 3.9.25 without executing imports. The old
127-file archive and manifest reverified with their original hashes unchanged.

The refreshed ZIP captures bytes before this status/ledger update: later
documentation changes do not rewrite it or invalidate capture-time integrity.
Receipts are `run_attempts/2026-10-09_source_bundle_v2_component_receipt.json`
and `run_attempts/2026-10-09_source_bundle_v2_syntax.json`. This does not package
frozen inputs, scores, trained weights, raw datasets or PDFs, run a full suite,
establish experimental reproduction, authorize publication or complete the goal.
Scientific completion and the documented end-to-end smoke remain pending.

## Pre-results working manuscript PDF

`analysis/render_manuscript_draft.py` creates an explicitly labelled working
manuscript from the existing frozen methods and selected-prior review. It
re-verifies all eighteen full-budget models and recomputes the training table
in memory, checks the bounded cohort/live/process/cache/city audit evidence,
and validates the source convergence image/QA lineage. It imports no Torch,
NumPy or Matplotlib and performs no training, inference or official scoring.
Existing/symlink/out-of-scope output targets are refused before rendering.
If any complete controlled score report appears, it refuses the pre-results
recipe rather than silently retaining a stale no-results manuscript.

The accepted output is
`output/pdf/navsim_controlled_study_draft_20261009_r2.pdf` (seven A4 pages).
It covers introduction, selected related work, controlled methods, audited
training diagnostics/convergence, technical evidence, conditional rules,
limitations and ten version-bound references. All pages retain a visible
working-draft/controlled-results-pending warning. No controlled PDMS results,
mechanism superiority, new architecture, acceptance or deployment safety claim
is inserted. Four stale bibliography reading-scope notes were brought into
agreement with the existing bounded review receipts, without changing any
scientific input, model, score, source or frozen gate.

The first PDF is preserved but not delivered: visual inspection found the
convergence image horizontally stretched/cropped because lazy image sizing
reset its width. Explicit source-image dimensions corrected the placement.
All seven revision-2 pages were rendered at 150 dpi and visually inspected;
text/image bounds, figure aspect ratio, training values and ten primary reference
URLs passed separate checks. Poppler's `No display font for Symbol` warning is
retained; no missing glyphs or clipping were visible in the accepted pages.

The revision-2 render receipt and `.visual_qa.json` beside the PDF bind its bytes,
all inspected page PNGs and audit/source/model lineage. Later status/ledger
updates are bookkeeping after capture, not changes to that PDF or its bound
inputs. This closes working-draft PDF layout QA only, not final manuscript,
scientific results, updated full-suite testing, final citation verification or
documented end-to-end experimental reproduction. Original cache/scoring
controllers remain in the user-authorized resource wait.

## Required remaining work

1. Keep the verified 18-model core artifacts and sampled live parity/recovery
   evidence intact; perform final preservation/provenance checks after experiments.
2. Finish complete validation/confirmation caching and separately exercise cache
   corruption/recovery; preserve historical caches and disclose failures.
3. Score validation, frozen confirmation and exploratory navtest; audit token
   coverage, exceptions, provenance and all submetrics.
4. Produce paired per-seed effects, session-cluster intervals, multiplicity-aware
   primary comparisons, scenario effects and parameter/latency/convergence tables.
5. Apply the planned mechanism gate. Where informative, run the 12 additional
   fixed-step learning-curve models and equal-epoch sensitivity; do not silently
   promote smoke models or skip a failed experiment.
6. Generate the scientific figures and deterministically selected failure cases.
7. Review closest related work in full; conditionally execute the v2 cross-protocol
   extension after the core evidence/feasibility gate, with separate environment
   and caches. Do not relabel it full closed-loop safety validation.
8. Write the evidence-grounded manuscript and reproducibility package, then audit
   every paper-plan requirement. No acceptance or unmeasured benefit is promised.

## Reproduction commands

The following are historical initial-execution examples, not instructions to
rerun against this bound experiment. Several default audit outputs overwrite
receipts and preparation regenerates its report; preserve the originals while
controllers are live. Use [the current guide](REPRODUCIBILITY.md) for independent
fresh-output checks, the guarded full suite and the local conditional command
index. It also discloses that a custom preparation `--out` does not redirect
controllers whose protocol path is fixed. No new-host end-to-end run is claimed.

Historical examples from the repository with the NAVSIM environment:

```powershell
. E:\navsim_workspace\env.ps1
& $env:NAVSIM_PYTHON analysis\audit_features.py --threads 2
& $env:NAVSIM_PYTHON analysis\validate_score_path.py --scenes 32
& $env:NAVSIM_PYTHON analysis\prepare_mechanism.py
& $env:NAVSIM_PYTHON analysis\run_mechanism_core.py --threads 2
```

Do not launch a second core runner: its OS-level lock prevents duplicate work.
Do not launch cache construction concurrently with full training on this 16 GB
machine. After core training, use `cache_mechanism.py --split validation`, then
`--split confirmation`; both share the new versioned cache and preserve the
historical navtest cache. Prediction/score tools have explicit run/output paths.

The reviewed isolated cache/score route is recorded below. Both matching
controllers are currently live; do not launch duplicates after a polling timeout:

```powershell
& $env:NAVSIM_PYTHON analysis\resume_mechanism_cache.py
# Only after both complete cache receipts; --wait-for-cache can queue on the lock.
& $env:NAVSIM_PYTHON analysis\run_mechanism_scoring.py --wait-for-cache
```

Do not launch a second evaluator; its cross-process lock prevents duplicates.
The scorer waits on the core lock, verifies actual cache/core completion and checks
source/policy hashes before each child stage. State JSON alone is not liveness.
Do not resume the old monolithic raw-audit route as a workaround for low RAM.
City training/evaluation commands are separate and fail closed before their
actual-score gate and prerequisite completion; they have not been launched.
