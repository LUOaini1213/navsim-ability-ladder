# Controlled-study reproducibility instructions

This document describes current local commands. It is not yet a complete final
release: scoring, applicable extensions, scientific figures, manuscript and the
row-by-row [completion audit](COMPLETION_REQUIREMENTS.md) remain outstanding.
No remote publication, benchmark submission or dataset redistribution is implied.

## Verified environment and sources

The current experiment uses Windows, Python 3.9.25, CPU Torch 2.0.1+cpu and
NumPy 1.23.4. Historical environment comments are not its dependency lock.
NAVSIM's local base commit is `3e8291bfa89ff247231e0227778840cd0a036896`;
the project base is `75231f25d1118433fd01b767d25ed880fa9ea3e0` plus the
current working changes. `results/mechanism_v2/runtime_snapshot/runtime.json`
records installed dependency versions, current tracked upstream file hashes,
canonical-agent hashes and resource measurements. It was captured during caching,
not before all historical runs, and does not establish retrospective immutability.

The local devkit has pre-existing Windows dataloader changes and historical
untracked agents. Preserve them. The current runtime overlays this repository's
canonical agents without modifying those copies. The runtime snapshot includes
the tracked dataloader patch; applying it to a new checkout must be verified,
not assumed to work on other operating systems.

Repository code has its own MIT license; NAVSIM carries Apache-2.0 text locally.
Those licenses do not cover nuPlan/OpenScene/map redistribution or copied papers.
Obtain datasets through their providers and retain third-party notices; do not
bundle raw logs, maps, sensors, caches or reviewed-paper PDFs in a public release.
Model/data-derived artifact redistribution requires a separate terms review.

## Local commands and resource exclusion

Run in the repository, using the NAVSIM environment, not an arbitrary document
runtime. Keep the 16-GB-machine heavy jobs serial. Do not duplicate an already
live evaluator/curve runner and do not terminate unrelated user jobs.

```powershell
$env:NAVSIM_WORKSPACE = 'E:\navsim_workspace'
. .\scripts\env.ps1
```

This setup block only sets the shell environment. Do not rerun default-output
snapshot/audit/summary/render commands as read-only checks: several overwrite
receipts or figures that existing controllers bind by hash. While the current
cache/scoring controllers are live, preserve their runtime/audit files even
when they have no child. Do not launch the commands below during that wait.

After all such controllers are terminal, with sufficient RAM and no conflicting
heavy job, put independent diagnostic outputs in a **fresh** unbound directory:

```powershell
$reproCheckRoot = 'results\reproduction_checks\FRESH_NAME'
if (Test-Path -LiteralPath $reproCheckRoot) { throw 'Choose a fresh check directory; preserve prior outputs' }
& $env:NAVSIM_PYTHON -X utf8 analysis\snapshot_mechanism_runtime.py --out "$reproCheckRoot\runtime"
& $env:NAVSIM_PYTHON -X utf8 analysis\verify_mechanism_core.py --out "$reproCheckRoot\core_manifest.json"
& $env:NAVSIM_PYTHON -X utf8 analysis\summarize_mechanism_training.py --out "$reproCheckRoot\training_diagnostics"
& $env:NAVSIM_PYTHON -X utf8 analysis\plot_mechanism.py --training-only --out "$reproCheckRoot\training_figures"
```

The first three checks do not import Torch; the renderer imports the graphics
runtime and still requires sufficient headroom. New snapshot/check outputs are
observations now, not replacements for original hash-bound evidence. Rendering
requires separate PNG/SVG visual QA. Do not point `--out` back at bound folders.

Run the updated full suite only with sufficient RAM, after the current
controllers finish, under the shared heavy-job lock:

```powershell
& $env:NAVSIM_PYTHON -X utf8 analysis\run_guarded_command.py --receipt results\mechanism_v2\run_attempts\full_suite_FRESH_NAME.json -- $env:NAVSIM_PYTHON -X utf8 -B -m unittest discover -s tests -v
```

This wrapper refuses a busy lock, preserves existing receipt/log paths and has
the generic RAM watchdog, not the cache-specific 6,000-MB guarantee. Some tests
import the ML runtime; do not use this example to bypass the user-selected RAM
wait. A successful suite is not a full scientific evaluation or experimental
smoke. Never wrap a lock-owning scientific controller in this wrapper.

On a new compatible host, obtain the appropriate raw logs/maps and build the
feature archives with the documented extraction path. Then freeze inputs in a
**new output folder** before training or reading outcomes:

```powershell
& $env:NAVSIM_PYTHON analysis\prepare_mechanism.py --out results\reproduction_protocol
```

The original protocol contains absolute feature paths. Changing paths changes its
byte hash: never rewrite the original protocol/checkpoint receipts just to make
them appear portable. New-host reproduction must explicitly document path/hash
differences and verify identical semantic split/session/fraction/donor arrays,
feature contents, settings and last-checkpoint policy. Existing strict model
receipts must not be falsely rebound to a different protocol.

The `--out results\reproduction_protocol` example freezes inputs only. It does
not redirect the core/cache/scoring or seed controllers, which still target
this checkout's `results/mechanism_v2/protocol`. The core runner has no
`--protocol` flag. A new reproduction needs an isolated checkout/output runtime
with its own canonical protocol location, or a separately implemented and tested
relocation path. Changing `--workspace` alone does not make these controllers a
portable new-protocol pipeline. No new-host end-to-end execution is claimed.

For the original local frozen protocol, the experiment commands are:

```powershell
& $env:NAVSIM_PYTHON analysis\run_mechanism_core.py --threads 2
& $env:NAVSIM_PYTHON analysis\run_mechanism_evaluation.py --threads 2
```

These are the historical core/fresh-audit route, not commands to rerun against
the currently bound experiment. In particular, do not restart the monolithic
raw-audit evaluator after the reviewed memory stop. The current local recovery
route is the existing cache-only controller followed by the existing scorer.

The evaluation controller acquires the heavy-job lock, verifies all 18 core runs,
audits inputs, exercises real interrupt recovery/live parity, smoke-tests caching,
then evaluates validation, confirmation and exploratory navtest serially. The
curve runner waits for the same lock, recomputes the frozen validation gate,
checks hash-bound actual score CSVs, and trains 24 smaller-data jobs only if
required (12 fixed-update, 12 equal-epoch). Full-data M1/M4 seeds are reused.
Curve-runner success covers training only. The separate registered evaluator is
`analysis/run_mechanism_curve_evaluation.py`: it verifies all 24 follow-ups plus
six full-data anchors before confirmation, uses separate curve prediction/score
folders, and produces paired budget/fraction sensitivities. Its implementation
does not prove actual execution or figure QA. The city and curve evaluators now
include `plot_followups.py`, which rebuilds statistics from full model/gate/
prediction/score audits and refuses a stale or modified report. City rebuilds
also require the actual frozen city execution trigger; the 12-cell exporter is
not a workaround for the 18-core-plus-six-auxiliary confirmation guard.
Neither runner completes
the whole paper goal by itself.

## Local execution index and conditional follow-up commands

This index identifies inspected local entry points and required evidence. It is
not an instruction to launch all rows, proof of execution, or a final portable
one-run reproduction recipe. Keep the heavy jobs serial and never recreate a
controller merely because its observation timed out. A wait flag queues on the
lock; it does not deduplicate controllers or authorize source/receipt changes.

| stage / entry point | evidence before launch | successful scope, not current completion |
|---|---|---|
| `resume_mechanism_cache.py` | retained successful audits, original failed attempt, unchanged 18-model/core/runtime bindings; no live duplicate | both full validation/confirmation cache receipts; no scores |
| `run_mechanism_scoring.py --wait-for-cache` | actual completed cache attempt and all eighteen eligible models | 54 exports, three official score batches, reports/cases/gates/figures; visual QA remains |
| `run_mechanism_curves.py` | audited actual validation CSVs and positive frozen learning-curve gate; execute after core scoring | 24 small-data fits, or explicit `not_triggered`; no official curve scores |
| `run_mechanism_curve_evaluation.py` | triggered gate, all 24 eligible follow-ups plus six full-data anchors, complete split caches | three audited thirty-run curve batches/reports/figures; visual QA remains |
| `run_city_training.py` | frozen city inputs/audit, unchanged full-data anchors and actual validation-driven city gate | six auxiliary fits, or clean-alignment claim withheld; no city scores |
| `run_city_evaluation.py` | actual `core_scoring_completed` and `six_city_models_verified` attempts, matching trigger and all model audits | 36 common-cohort exports, three twelve-cell batches/reports/figures; visual QA remains |
| `run_seed_training.py` | complete validation/confirmation actual CSV effects and registered paired-seed trigger | twelve added fits, or explicit no-trigger with no new fits |
| `run_seed_evaluation.py` | actual completed core-scoring and twelve-added-model attempts; all thirty models audited | 36 added exports, three separate twelve-run batches and five-seed reports; no figures |
| `run_seed_figures.py` | actual `seed_evaluation_completed` attempt and source-rebuilt thirty-model reports | four PNG/SVG pairs per split; independent visual QA remains |

The recorded current recovery commands below are already represented by live
controllers. Do not execute another copy while either matching controller exists:

```powershell
& $env:NAVSIM_PYTHON analysis\resume_mechanism_cache.py
& $env:NAVSIM_PYTHON analysis\run_mechanism_scoring.py --wait-for-cache
```

For the applicable learning-curve branch, after the existing core scorer finishes:

```powershell
& $env:NAVSIM_PYTHON analysis\run_mechanism_curves.py --workspace E:\navsim_workspace --threads 2
# Only after a triggered branch verifies all twenty-four registered-budget artifacts.
& $env:NAVSIM_PYTHON analysis\run_mechanism_curve_evaluation.py --workspace E:\navsim_workspace
```

For the applicable same-city branch, after core scoring and its frozen input audit:

```powershell
& $env:NAVSIM_PYTHON analysis\run_city_training.py --workspace E:\navsim_workspace
# Only after the actual trigger and six auxiliary model receipts pass.
& $env:NAVSIM_PYTHON analysis\run_city_evaluation.py --workspace E:\navsim_workspace
```

If a training branch records no trigger, retain that gate and its claim/omission
disclosure; do not run the positive-branch evaluator with missing models. The
five-seed training/evaluation/graphics commands and safeguards are detailed below.
These local entries do not implement the benchmark-v2 migration described in
`CROSS_PROTOCOL_FEASIBILITY.md`. That branch needs its actual outcome/feasibility
gate, independent environment and real structured-input/synthetic-state smoke.
The final manuscript, final artifact/license audit and documented end-to-end
experimental smoke still need their own actual deliverables and commands.

The five-seed sensitivity extension has a separate prospective registration at
`results/mechanism_v2/seed_protocol/registration.json`. Its guarded training entry
may be invoked only after complete core validation and confirmation score
evidence; it recomputes the actual paired CSV effects and requires the hash-bound
`gate_confirmation.json` before creating any training protocol/donor copy:

```powershell
& $env:NAVSIM_PYTHON analysis\run_seed_training.py
```

This command refuses a busy heavy-job lock by default. Explicit
`--wait-for-core-scoring` can queue, but do not create another controller while a
matching task is live. A no-trigger result produces no added fits. A triggered
run preserves the original protocol and donors, creates a separate `seed_protocol/active`
copy and fits seeds 3/4 for all six conditions under
`E:/navsim_workspace/exp/mechanism_v2/seed_models`, reusing the unchanged trainer.
All twelve full-budget artifacts and original-normalizer parity must pass before
an eventual evaluator accesses their confirmation predictions.

This new training entry requires 6,000 MB continuously for 30 seconds and polls
every 0.5 seconds, stopping only its owned child below 2,000 MB (1,500 MB emergency).
It never automatically retries a memory interruption or ordinary failure.
Inspect and preserve the terminal receipt before an explicit checkpoint-resuming
invocation; changed or partial protocol copies are not silently repaired.
The actual local scientific runner has not been launched. Prediction, official
scoring and five-seed analysis are not performed by this training entry. Their
separate implemented evaluator requires completed core-scoring and seed-training
attempts plus an independent all-thirty-artifact check:

```powershell
& $env:NAVSIM_PYTHON analysis\run_seed_evaluation.py
```

Default lock acquisition is non-waiting; optional `--wait` does not authorize a
duplicate live evaluator. The script makes 36 added-seed exports and scores only
seeds 3/4 in separate twelve-run batches at `results/mechanism_v2/seed_scores`.
It must not mix original and extended training-protocol hashes in the official
scorer or falsely rebind old prediction metadata. `analyze_seed.py` reads the
unchanged original eighteen-run archive and added twelve-run archive separately,
verifies exact cohort/provenance/all-seven-metric coverage and writes
`results/mechanism_v2/seed_analysis` with all five paired seeds and fixed 10,000
session/log bootstrap draws. The original three-seed core gate remains unchanged.

The new evaluator inherits the conservative 6,000-MB/no-automatic-retry policy.
It has not actually run. Fifty scoped seed-component tests, including small NPZ
and CSV fixtures and a synthetic thirty-run report with a mocked model-audit
layer, are not actual inference, official-engine evaluation, full experimental
reproduction or the updated full-suite result. Actual applicable extension fits,
scores, reports, scientific figure rendering/visual QA and final claim audit
remain necessary. Five-seed reports are outcome-gated sensitivity results, not
fresh independent confirmation or uncertainty over a population of new fits.

## Five-seed sensitivity graphics

After an actual triggered extension finishes all thirty-model audited reports,
use the dedicated guarded graphics runner (not a concurrent direct renderer):

```powershell
& $env:NAVSIM_PYTHON analysis\run_seed_figures.py --workspace E:\navsim_workspace
```

It requires `seed_evaluation_completed` with its real attempt/source bindings,
holds the same heavy-job lock and serializes validation, confirmation and
exploratory navtest. Each report is rebuilt from actual model, prediction and
score artifacts, with original and added training-protocol identities unchanged.
Launch requires 6,000 MB stable for 30 seconds; an owned child stops below
2,000 MB and is not automatically retried. Changed/partial figures are preserved;
choose a fresh `--out-root` version for a later explicit invocation.

Each split produces four PNG/SVG pairs: all-seven contrast forest with five
seed points, descriptive original-three/added-two means, input strata and
seven-submetric heatmap. The report remains an outcome-gated sensitivity study,
not fresh independent confirmation; session intervals condition on fitted seeds.
Empty and one-session strata are retained. Subset means have no new confidence
intervals, and DDC remains diagnostic-only in v1. All PNGs and independently
rasterized SVGs still need visual inspection and hash-bound QA.

For layout testing only, when at least 2,000 MB is available:

```powershell
& $env:NAVSIM_PYTHON analysis\plot_seed.py --layout-smoke --out tmp\seed_layout_FRESH_NAME
```

This mode rejects scientific reports and destinations outside repo `tmp`.
At this update no actual five-seed report or graphics run exists. The 23 new
and eight regression tests use synthetic/mock inputs and byte placeholders,
not rendered images, score-engine execution or full experimental reproduction.

## Results, failure recovery and boundaries

- Trained models/logs/predictions/cache: `E:/navsim_workspace/exp/mechanism_v2`.
- Immutable core inputs/policies: `results/mechanism_v2/protocol`.
- Actual audits: `results/audit_v2`; driving scores/analyses/cases are separate
  subfolders under `results/mechanism_v2` when completed.
- Training diagnostics: `results/mechanism_v2/training_diagnostics`, with all
  individual seeds and checkpoint/history hashes; ADE is not PDMS.
- Resume only matching existing settings/source hashes. A changed wrapper or
  policy requires a new output version rather than overwriting score provenance.
- SQLite scoring recovery and cache validity checks do not waive complete aligned
  scene coverage or failure-attempt disclosure. Failed stages must be inspected.
- A state JSON or stale PID is not evidence of a live process. Check the actual
  process/tool session before retrying. Do not duplicate a process after an
  observation timeout.
- The updated core evaluator/curve runners require 2,400 MB continuously for
  20 seconds before launch and inspect RAM every five seconds during a child.
  Below 512 MB once or below 1,000 MB twice stops only the owned child tree.
  Recorded memory interruptions may resume only supported cache/atomic-training/
  SQLite stages; ordinary failed commands are not silently retried or skipped.
  Preserve terminal attempt receipts before restarting a controller.
- For individual new checks, `analysis/run_guarded_command.py --receipt NEW.json
  -- EXECUTABLE ARGS...` holds the same heavy-job lock, logs child output and
  refuses to overwrite an old receipt/log. Do not wrap a lock-owning runner in
  this command; that would try to acquire the same lock twice.
- After the reviewed memory stop, `analysis/resume_mechanism_cache.py` can resume
  cache work only. It requires the retained original successful audit stages,
  unchanged sources/policies, actual passed audit receipts and the upstream
  runtime baseline. Its reused-audit timing is explicit, all attempts are saved,
  and no scoring/training or project completion follows from cache success.
  It will wait without a heavy child until safe headroom returns. Do not rerun a
  hash-bound audit or modify a bound source while this controller owns the lock.
  Its revised cache-specific safeguard requires 6,000 MB continuously for
  30 seconds, polls every 0.5 seconds, stops an owned child below 2,000 MB, and
  disables automatic retry after memory interruption. The original lower generic
  safeguards above are not permission to launch this high-memory raw cache stage.
  A fresh run cannot invent the reused successful audit receipts: complete the
  original input/recovery/live audits on a sufficiently provisioned machine first.
  For the already-audited local state, use the isolated cache controller and then
  `run_mechanism_scoring.py --wait-for-cache`, not another monolithic raw audit.
- Synthetic plot-layout smoke is visibly labelled and scientifically ineligible.
  New scientific PNG/SVG figures and diagnostic cases require visual inspection.
- Same-city and five-seed claims require their separate registered extensions;
  benchmark-v2 execution requires its feasibility gate. Omitted work needs a
  recorded negative gate/unsupported-feasibility reason, not an invented result.

The final release still needs an exact command index, a hash-manifest package,
one-run documented smoke, preserved historical-artifact verification and final
dependency/license checks. These tasks remain on the completion ledger.

## Follow-up graphics and vector QA

For a byte-exact local source/instruction snapshot, without raw data, model
weights, caches, paper PDFs or score archives:

```powershell
& $env:NAVSIM_PYTHON analysis\repro_source_bundle.py build --out results\repro_snapshots\FRESH_NAME
& $env:NAVSIM_PYTHON analysis\repro_source_bundle.py verify results\repro_snapshots\FRESH_NAME
```

The destination must be a fresh child directory; existing/failed attempts are
preserved. The source ZIP has deterministic ordering/timestamps and an external
per-file SHA-256/size manifest. Verification streams members without extracting
untrusted paths and rejects changed, extra, duplicate or oversized members.
New v2 snapshots include all currently allowlisted source/test files, including
the five-seed components, and require the single public bibliography file
`docs/reference_metadata_2026-10-09.json`. Other JSON files are not allowed.
The verifier still accepts unchanged historical v1 snapshots under their original
source-only allowlist; it does not rewrite them or imply they contain newer code.
Every snapshot records bytes at capture, not the evolving worktree afterward.
This tool validates source bytes, not experiment semantics, authenticity or data
licenses. It does not constitute the final package: frozen input lineage,
per-scene results, artifact redistribution review, final manuscript and an actual
documented end-to-end experimental smoke remain required. Nothing is uploaded.

Scientific city figures comprise paired alignment/encoder contrasts, overlapping
input strata and seven submetrics. Curves separately show fixed updates and equal
epochs, the paired encoder effect, four small-minus-full sensitivities and all
seven submetrics. The 100% anchor is reused, not an independent epoch-budget run.
Fitted-seed-conditional original-session intervals and auxiliary multiplicity
boundaries are explicit. DDC is only diagnostic in the pinned v1 score.

For layout testing only, after safe RAM is available:

```powershell
& $env:NAVSIM_PYTHON analysis\plot_followups.py --kind city --layout-smoke --out tmp\city_layout
& $env:NAVSIM_PYTHON analysis\plot_followups.py --kind curves --layout-smoke --out tmp\curve_layout
```

Smoke destinations must be inside this repository's `tmp`, cannot include
scientific inputs and are stamped ineligible. They do not fulfill scientific
figure requirements. Production requires a report, its complete score archive
and, for city figures, its exact prediction folder; the registered 10,000-draw
report is rebuilt before rendering. Owned partial renders may resume with exactly
matching settings; different source/report bindings require a new output version.

The real core convergence SVG was independently rasterized using
`analysis/render_svg_qa.cjs` with installed sharp 0.35.5/libvips 8.18.7/librsvg
2.63.2 and visually inspected. The scientific SVG and render receipt were not
modified. `vector_visual_qa.json` binds the inspected raster and independent
renderer receipt; this closes training-vector layout QA only. Reproduction takes
four explicit arguments: source SVG, fresh QA PNG, its scientific render receipt
and an installed sharp `node_modules` directory. This optional QA renderer is not
a Torch/NAVSIM experimental dependency. Node tests require
`SVG_QA_NODE_MODULES` to exercise the real renderer rather than skip it.

## Pre-results working manuscript export (document runtime only)

The current seven-page working PDF uses ReportLab 4.4.9 and pypdf 6.10.0 from
the inspected bundled document runtime; these are not NAVSIM training/scoring
dependencies. Obtain that runtime path from workspace dependencies on another
host. For this host, the following command was actually used with a fresh output:

```powershell
$draftDocumentPython = 'C:\Users\<user>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $draftDocumentPython -X utf8 -B analysis\render_manuscript_draft.py --out output\pdf\navsim_controlled_study_draft_20261009_r2.pdf
```

Do not rerun against the existing PDF: use a fresh output name and preserve
failed attempts. The renderer re-audits existing artifacts and rejects any
complete controlled report until the manuscript is revised to incorporate it.
It reads hashes/JSON and uses no Torch, NumPy, Matplotlib, training, inference
or scoring engine. This document export is not a shortcut around the scientific
RAM guard or a portable experimental pipeline.

The first PDF's image placement failed visual QA and was preserved. Revision 2
has all seven page PNGs inspected; its separate `.visual_qa.json` binds page
hashes, image bounds/aspect ratio, training-table values and ten bibliography
URLs. Poppler emitted a Symbol-display-font warning; accepted page glyphs are
readable. Reproduce PDF visual QA with Poppler and pdfplumber as document tools,
not scientific model checks. The draft contains no complete controlled PDMS
results and is not a completed, submitted or accepted paper. Final scientific
results, claim/citation audit and the end-to-end experimental smoke remain pending.
