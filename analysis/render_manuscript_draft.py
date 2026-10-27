"""Render a source-grounded working manuscript, never a finished score paper.

Only stdlib artifact audits and the document runtime are used. No model inference,
scoring, training, protocol mutation or archive extraction is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from summarize_mechanism_training import summarize
from verify_mechanism_core import CONFIGS, ROOT, digest, verify


def inspect_inputs(workspace):
    results = ROOT / "results/mechanism_v2"
    protocol_path = results / "protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    core = verify(workspace / "exp/mechanism_v2/models", protocol_path)
    summary = summarize(core, workspace / "exp/mechanism_v2/models")
    stored_path = results / "training_diagnostics/summary.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    if any(stored.get(key) != value for key, value in summary.items()):
        raise ValueError("Training table differs from current audited full-budget artifacts")
    if core["n_runs"] != 18 or core["optimizer_steps_per_run"] != 13000:
        raise ValueError("This draft describes the registered eighteen-model budget only")
    for split in ("validation", "confirmation", "navtest"):
        if (results / "analysis" / split / "report.json").exists():
            raise ValueError("Controlled report now exists: revise the manuscript instead of reusing the pre-results draft")
    paths = [protocol_path, protocol_path.parent / "donors.npz", stored_path,
             ROOT / "docs/CONTROLLED_METHODS.md", ROOT / "docs/CLOSEST_PRIOR_REVIEW.md",
             ROOT / "docs/reference_metadata_2026-10-09.json", ROOT / "docs/references.bib",
             ROOT / "docs/COMPLETION_REQUIREMENTS.md", protocol_path.parent / "diagnostics.json",
             protocol_path.parent / "followup_policy.json", results / "city_protocol/execution_policy.json",
             results / "seed_protocol/registration.json", results / "runtime_snapshot/runtime.json",
             ROOT / "analysis/verify_mechanism_core.py", ROOT / "analysis/summarize_mechanism_training.py",
             Path(__file__)]
    audits = {}
    for name in ("input_controls", "process_recovery", "mechanism_live", "cache_recovery", "city_inputs"):
        path = ROOT / "results/audit_v2" / (name + ".json")
        paths.append(path)
        audits[name] = json.loads(path.read_text(encoding="utf-8"))
        if audits[name].get("passed") is not True:
            raise ValueError("Required technical audit does not pass: " + name)
    inputs = audits["input_controls"]
    if (inputs["protocol_sha256"] != core["protocol_sha256"]
        or inputs["donors_sha256"] != core["donors_sha256"] or not inputs["cohort_enumerated"]):
        raise ValueError("Input audit is not for the current core")
    for split, expected in (("navtrain", 103288), ("navtest", 12146)):
        row = inputs["cohorts"][split]
        if (row["available_selected_raw_scenes"] != expected or row["feature_eligible_scenes"] != expected
            or row["selected_raw_scenes_without_features"] or row["missing_logs"]):
            raise ValueError("Cohort coverage no longer supports the draft")
    recovery = audits["process_recovery"]
    if (recovery["resumed_from_step"] != 325 or recovery["final_step"] != 1000
        or not all(recovery["checks"].values())):
        raise ValueError("Interrupted-process evidence differs from the described smoke")
    live = audits["mechanism_live"]["ledger"]
    if (len(live) != 64 or len({row["token"] for row in live}) != 16
        or any(not row["passed"] or row["diagnostic"]["used_fallback"] for row in live)):
        raise ValueError("Sampled parity evidence no longer matches the draft")
    fault_rows = audits["cache_recovery"]["attempts"]
    if (len(fault_rows) != 3 or audits["cache_recovery"]["historical_and_full_cache_roots_modified"]
        or audits["cache_recovery"]["scientific_result_eligible"]
        or any(not row["baseline_byte_equal"] or row["maximum_official_submetric_error"] != 0 for row in fault_rows)):
        raise ValueError("Cache recovery evidence differs from the bounded three-fault exercise")
    metadata = json.loads((ROOT / "docs/reference_metadata_2026-10-09.json").read_text(encoding="utf-8"))
    if len(metadata["references"]) != 10 or any(row["review_scope"] != "selected_text_review" for row in metadata["references"]):
        raise ValueError("Bibliography review scope changed")
    image_folder = results / "figures/training"
    image_receipt = json.loads((image_folder / "receipt.json").read_text(encoding="utf-8"))
    image_qa = json.loads((image_folder / "visual_qa.json").read_text(encoding="utf-8"))
    vector_qa = json.loads((image_folder / "vector_visual_qa.json").read_text(encoding="utf-8"))
    image = image_folder / "core_convergence.png"
    if (digest(image) != image_receipt["figures"][image.name]
        or digest(image) != image_qa["inspected_png_sha256"]
        or digest(image_folder / "core_convergence.svg") != vector_qa["svg_sha256"]
        or image_receipt["checkpoint_sha256"] != summary["checkpoint_sha256"]
        or image_receipt["history_sha256"] != summary["history_sha256"]):
        raise ValueError("Audited convergence graphic changed")
    paths.extend(image_folder / name for name in ("receipt.json", "visual_qa.json", "vector_visual_qa.json",
                                                 "core_convergence.png", "core_convergence.svg"))
    cache_status = json.loads((results / "cache_resume_status.json").read_text(encoding="utf-8"))
    for name, sha in cache_status["frozen_sha256"].items():
        if digest(name) != sha:
            raise ValueError("Live scientific binding changed: " + name)
    return {"core": core, "summary": summary, "protocol": protocol, "audits": audits,
            "metadata": metadata, "image": image, "max_live_error": max(row["max_error"] for row in live),
            "source_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
            "scientific_binding_count": len(cache_status["frozen_sha256"])}


def render(output, facts):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    width = A4[0] - 84
    styles = {
        "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=13.7,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "heading": ParagraphStyle("heading", fontName="Times-Bold", fontSize=13.5, leading=17,
                                  spaceAfter=9, spaceBefore=5, keepWithNext=True),
        "small": ParagraphStyle("small", fontName="Times-Roman", fontSize=9.1, leading=11.7, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=11.5),
        "numeric": ParagraphStyle("numeric", fontName="Times-Roman", fontSize=8.4, leading=10.5),
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=20, leading=24,
                                alignment=TA_CENTER, spaceAfter=12),
        "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=9, leading=12,
                               alignment=TA_CENTER, spaceAfter=10),
        "warning": ParagraphStyle("warning", fontName="Helvetica-Bold", fontSize=9.8, leading=13,
                                  textColor=colors.HexColor("#9D3027"), spaceAfter=8),
    }
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        p(text, "heading")

    def new_page():
        story.append(PageBreak())

    def table(rows, columns, numeric=False):
        cell = styles["numeric" if numeric else "cell"]
        grid = [[Paragraph(str(item), cell) for item in row] for row in rows]
        item = Table(grid, colWidths=columns, hAlign="LEFT", repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#536472")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor("#536472")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([item, Spacer(1, 8)])

    p("When Does Current-Object Information Help<br/>Imitation-Based Planning?", "title")
    p("A controlled NAVSIM study | Wenjie Luo | 9 October 2026", "meta")
    p("WORKING MANUSCRIPT: CONTROLLED DRIVING-SCORE RESULTS ARE PENDING", "warning")
    p("This draft reports verified design, training diagnostics and bounded technical audits only. "
      "It is not a completed scientific paper, submission, acceptance claim or deployment-safety assessment.", "small")
    heading("Abstract")
    p("Additional scene information need not yield monotonic improvements in a finite-budget imitation planner. "
      "We study selected current-object annotations under a fixed ego-state and centerline context, separating "
      "sorted concatenation, masked set encoding, a low-dimensional lead summary and cross-session donor controls. "
      "The frozen core comprises six conditions and three paired seeds on session-disjoint training, validation "
      "and confirmation partitions. All eighteen full-budget models have passed artifact checks. We disclose "
      "cross-city donor mixing, the confirmation set's historical use, approximate fitted-seed-conditional "
      "uncertainty, and the privileged-input/nonreactive evaluation boundary. This draft includes audited "
      "imitation diagnostics, not complete controlled PDMS comparisons. It therefore reaches no conclusion "
      "about an object's driving-score benefit, a supported mechanism, general perception redundancy or safety.")
    heading("1. Introduction and research question")
    p("The archived ability ladder motivated a question rather than a new architecture: under what "
      "representation, data budget and evaluation conditions does selected current-object information improve "
      "planning? Historical navtest observations informed this design and remain exploratory. A controlled "
      "comparison must distinguish useful information from added dimensions, normalization, capacity and "
      "optimization. We do not infer a planning effect from a lower trajectory-fitting loss.")
    p("Three competing explanations guide the design. Representation and optimization may make sorted "
      "object slots difficult to exploit. More complex inputs may require a different data or update budget. "
      "Finally, ego behavior and scene composition may make aggregate short-horizon scores insensitive to "
      "some current traffic cues. These explanations can coexist; none is presumed established.")
    p("The candidate contribution is a transparent controlled analysis of current-object alignment, encoding "
      "and data budget, with paired-seed/session reporting and explicit stopping rules. Generic input ablations, "
      "a set encoder or an added benchmark alone are not claimed novel. Negative or inconclusive outcomes remain "
      "valid technical-report outcomes after the applicable experiments are actually completed.")

    new_page()
    heading("2. Related work and claim distinction")
    p("<b>Centerline priors and imitation.</b> PDM establishes centerline-conditioned learning and separates "
      "forecasting from planning evaluation [1]. PlanTF studies feature reliance, augmentation and imitation "
      "gaps [4]. Our present-state, four-second privileged-input MLPs are not replications of their nuPlan "
      "settings. Centerline conditioning and the observation that additional features can be difficult to "
      "exploit are not standalone contributions here.")
    p("<b>Ego-state reliance and evaluation.</b> Ego Status examines weak exploitation of perception in "
      "open-loop end-to-end driving [5]. NAVSIM introduces scalable nonreactive simulation metrics and "
      "already includes input/data ablations and multiple seeds [2]. These motivate careful extraction, "
      "metric and subgroup checks, not a claim that perception is unnecessary. None of their published "
      "scores is a matched-budget comparison with this core.")
    p("<b>Current information versus static context.</b> Driving on Memory investigates replacing current "
      "camera observations with retrieved memory, and includes map/context and traffic analyses [6]. Our "
      "narrow distinction is separately trained current-object/donor conditions with fixed ego/centerline "
      "context. A map-strength narrative or another scenario breakdown does not by itself distinguish "
      "the present study from that work.")
    p("<b>Established set representations.</b> Deep Sets supplies a mature shared-element/commutative-pooling "
      "design [3]; Set Transformer uses attention-based set processing [8]. M4/M5 use a finite-width row-wise "
      "MLP and masked mean/max, not Set Transformer. Order/padding checks support tested implementation "
      "invariance, not universal approximation, learned interaction reasoning, superior sample efficiency "
      "or safety. M4-M1 changes an encoder bundle and does not isolate invariance as a causal mechanism.")
    p("<b>Beyond the pinned scoring protocol.</b> Pseudo-Simulation motivates testing synthetic states and "
      "limited reactive traffic in a distinct benchmark-v2 protocol [7]. Cross-benchmark correlation studies "
      "published paired method means [9]; its small, heterogeneous sample is not validation of our models. "
      "We retain the registered official score and all subscores rather than fitting a proxy or transferring "
      "external aggregate correlations into a causal claim.")
    p("<b>Input scope.</b> Traffic Element Awareness studies regulatory lights/signs across driving pipelines "
      "[10]. Our nearest-eight tracked annotations neither represent a full sensor stack nor cover all "
      "regulatory semantics. A result about this restricted branch cannot establish general perception "
      "redundancy. Author-reported ECCV acceptance is not an independently verified proceedings entry here.")
    p("All ten references have bounded selected-text reviews, with version and reading limits retained locally. "
      "Pseudo-Simulation's separate supplement was not read, Traffic Element Awareness was not read in all "
      "fifty pages, and no cited baseline was reproduced. This comparison is not an exhaustive novelty certificate.", "small")

    new_page()
    heading("3. Controlled protocol, inputs and target")
    splits = facts["protocol"]["split_counts"]
    table([["Partition", "Scenes", "Original sessions", "Role"]] +
          [[name.title(), f"{splits[name]['scenes']:,}", str(splits[name]["sessions"]), role]
           for name, role in (("train", "Optimization + normalization"), ("validation", "Diagnostics + follow-up gates"),
                              ("confirmation", "Current-model held-out confirmation"))] +
          [["Navtest", "12,146", "44", "Historical/exploratory evaluation"]], [81, 64, 103, width-248])
    p("The feature archives contain 103,288 navtrain scenes and 12,146 navtest scenes. Training/validation/"
      "confirmation are city-stratified and disjoint by original driving session, not merely segmented log. "
      "All 44 navtest sessions are disjoint from navtrain. Confirmation is excluded from current-model "
      "optimization, normalization and selection, but historical models saw those data; it is not a "
      "pristine never-touched external test. Compressed archives contain targets for all partitions; "
      "the current trainer indexes training/validation only.")
    p("Each core condition receives eight ego channels and twenty centerline samples over a 60 m lookahead "
      "(eighty position/orientation channels). The object branch keeps the nearest eight selected objects "
      "within 60 m: local position, absolute velocity in the ego basis, length/width, heading cosine/sine, "
      "vehicle/pedestrian indicators and a validity mask. The selected categories are vehicle, pedestrian, "
      "bicycle and generic_object. Bicycle/generic objects have neither type flag; generic_object is not "
      "an upstream moving-agent category. These are privileged annotations and map routes.")
    table([["Condition", "Object representation", "Independent lead summary"],
           ["M0", "None; object input ignored", "Ignored"],
           ["M1", "Sorted real eight-object concat + mask", "Ignored"],
           ["M2", "Matched cross-session donor concat", "Ignored"],
           ["M3", "None", "Current bumper gap + lead speed"],
           ["M4", "Real masked mean/max set encoder", "Ignored"],
           ["M5", "Same donors as M2; masked set encoder", "Ignored"]], [62, width-224, 162])
    p("The prediction is eight relative x/y/heading poses at 0.5 s intervals over four seconds. Training "
      "minimizes mean absolute error over twenty-four coordinates mixing metres and radians. It is not an "
      "angular geodesic or calibrated safety loss. Coordinate L1, raw-heading L1, ADE and FDE are reported "
      "separately; progress-only losses from the old ladder have different units.")

    new_page()
    heading("4. Optimization and donor-control qualifications")
    p("All conditions share three 512-wide ReLU pose-head layers. Set branches transform ten object channels "
      "with two 32-wide ReLU layers, then concatenate masked mean/max with the 88-channel context. Empty sets "
      "produce zero summaries and padding is excluded. The largest/smallest parameter ratio is "
      f"{facts['core']['parameter_ratio']:.7f}, within the registered 10% tolerance; effective capacity is not identical.")
    p("Normalization uses selected training scenes only. The first six continuous object channels are "
      "standardized over valid training objects pooled across slots; heading/type/mask channels are unscaled. "
      "Padding is zeroed after transformation and inactive branches are ignored. All full-data core models "
      "share a normalizer. Paired seeds 0/1/2 use Adam at 0.001, batch 256, no weight decay/dropout and the "
      "last checkpoint. Each run completes 13,000 updates, equivalent to forty virtual full-data epochs. "
      "Validation loss does not select a checkpoint.")
    p("Base donors remain within the same partition and come from another original session. Matching uses "
      "current command, speed bins at 2/5/10/20 m/s and object-count bins at 1/3/6; recorded relaxation "
      "proceeds through count, speed and command. A donor session is sampled before a scene. No future pose "
      "or actor annotation is a matching input. This alters scene consistency, not physically realizable traffic.")
    donor_rows = facts["audits"]["input_controls"]["donors"]
    p("The input audit finds same-city fractions of " + ", ".join(
        f"{name} {100*donor_rows[name]['same_city_fraction']:.1f}%" for name in ("train", "validation", "confirmation", "navtest")) +
      ". Thus base M1-M2 and M4-M5 combine alignment and geographical shifts. They cannot alone isolate "
      "current traffic reasoning or a clean causal alignment effect.")
    city = facts["audits"]["city_inputs"]["partitions"]
    p("The separately frozen same-city/different-session design preserves the full training cohort and "
      "normalizer, allowing the M1/M4 real-input anchors to be reused. Singapore has only one session in "
      "validation and confirmation and is excluded from their strict city control. Common coverage is "
      f"{city['validation']['n_scenes']:,}/{city['validation']['n_original_sessions']} validation scenes/sessions "
      f"and {city['confirmation']['n_scenes']:,}/{city['confirmation']['n_original_sessions']} confirmation "
      "scenes/sessions. Six auxiliary M2/M5 fits and matched-coverage scores remain outcome-gated and pending. "
      "Same-city swapping still need not respect routes or actor compatibility.")
    p("A same-scene reversal is separate from donor training: it retains the object set and diagnoses slot "
      "sensitivity at inference without retraining. The recorded check uses up to the first 512 fixed "
      "validation inputs; it is not all permutations, an object-counterfactual experiment or interaction proof.")
    heading("5. Scoring and paired uncertainty (planned, not completed)")
    p("Use the unchanged NAVSIM v1.1 PDM implementation and full scene coverage. Report PDMS, NC, DAC, TTC, "
      "EP, comfort and DDC. The pinned progress/TTC/comfort weights are 5/5/2 and DDC has zero weight, "
      "so DDC is diagnostic. Per-token official scores are averaged directly; a product of averaged "
      "subscores is not a substitute. Final invalid scores and failed recovery attempts are distinct disclosures.")
    p("Primary paired contrasts are M1-M0, M4-M1 and M4-M5. Report all seed effects, their scene-weighted "
      "mean, 10,000 whole-session paired bootstrap draws and log-cluster sensitivity. The three primary "
      "intervals use Bonferroni tail allocation, with approximate percentile coverage. Intervals condition "
      "on the fitted seeds, not new-model training uncertainty; fewer than two sessions yields no estimable "
      "population interval. Subgroups/secondary contrasts remain exploratory and no trajectory ensemble is scored.")

    new_page()
    heading("6. Verified evidence available in this draft")
    p("Current artifact verification confirms all eighteen last-checkpoint full-budget models, their "
      "source/protocol/donor identities, a shared full-data normalizer and the registered updates. The table "
      "below is regenerated from these audited artifacts, not hand-entered driving scores. Entries are "
      "three-seed mean [minimum, maximum], not confidence intervals or population-training uncertainty.")
    rows = [["Model", "Parameters", "ADE (m)", "xy L1 (m)", "heading L1 (rad)", "forward (ms)"]]
    def cell(values):
        return f"{values['mean']:.4f}<br/>[{values['min']:.4f}, {values['max']:.4f}]"
    for name in CONFIGS:
        row = facts["summary"]["summary"][name]
        rows.append([name, f"{row['parameters']:,}", cell(row["losses"]["ade_m"]), cell(row["losses"]["xy_l1_m"]),
                     cell(row["losses"]["heading_l1_rad"]), cell(row["diagnostics"]["cpu_single_scene_ms"])])
    table(rows, [39, 68, 99, 99, 99, width-404], numeric=True)
    p("Timing is one cached tensor scene: ten warm-ups and 200 neural forward passes using two CPU threads "
      "on a shared host. It excludes extraction, maps, sensors, transfer, I/O and simulation. It is not "
      "end-to-end latency or a real-time claim. Loss rankings in this table do not establish PDMS rankings.", "small")
    image_width, image_height = ImageReader(str(facts["image"])).getSize()
    image = Image(str(facts["image"]), width=width, height=width * image_height / image_width)
    story.append(image)
    p("<b>Figure 1.</b> Actual full-budget validation convergence. Lines/bands are mean and min/max across "
      "three fitted seeds, not confidence bands. Axes retain metre/radian units. The source PNG and an "
      "independently rasterized SVG were inspected; this page also needs PDF-level visual QA. Training "
      "convergence is not official driving-score or safety evidence.", "small")

    new_page()
    heading("7. Technical audit scope and applicable follow-ups")
    p("The selected-cohort audit enumerates all 103,288 navtrain and 12,146 navtest feature-eligible scenes "
      "on the available 1,192/136 logs, with no missing listed logs or selected-scene features. This does "
      "not reconstruct historical fallback reasons or cover every underlying frame. The current controller "
      "explicitly reuses that earlier audit rather than pretending it was frozen before historical training.")
    p(f"Live/cached parity covers sixteen deterministic navtest scenes and M0/M1/M3/M4 at seed zero: "
      f"sixty-four comparisons, maximum mixed-component error {facts['max_live_error']:.3g}, and no fallback "
      "in that sample. It does not cover all scenes, seeds or live donor behavior. An actual update-325 "
      "interruption/resume to 1,000 smoke updates exactly matched model, Adam, RNG, sampler and validation "
      "history. A separate three-fault cache exercise recovered byte-identical results with zero official "
      "submetric error. Neither smoke exercise is a full-budget scientific result or a completed full cache.")
    p("Numeric follow-up rules were bound after core training started but before complete controlled PDMS "
      "comparisons; they were not all specified in the initial prose plan. Any primary validation mean with "
      "absolute PDMS effect at least 0.01 triggers twelve nested 10/30% fixed-update fits and twelve equal-"
      "epoch fits for M1/M4. The full-data anchors are reused. A paired contrast with seed effects at or "
      "below -0.01 and at or above +0.01 in validation or confirmation requires registered paired seeds "
      "3/4 for a key claim, otherwise that claim is withheld.")
    p("The city execution rule was bound at 08:39:33 UTC on 9 October 2026: absolute validation M1-M2 or "
      "M4-M5 of at least 0.01 triggers six auxiliary fits. Without the control, clean-alignment claims "
      "are withheld. A benchmark-v2 feasibility attempt additionally requires the same primary contrast "
      "to meet the practical threshold in validation/confirmation, agree in direction with at least "
      "two of three seeds in each, and have both adjusted session intervals exclude zero in that direction. "
      "This authorizes an isolated attempt, not a v2 finding or full closed-loop validation.")
    p("Input-defined strata use frozen thresholds and overlap. The crossing proxy is current-velocity/"
      "straight-axis extrapolation, not an observed future cut-in or map-aware collision label. Cases use "
      "fixed mean-PDMS bins (positive at least 0.05, negative at most -0.05, neutral absolute effect at most "
      "0.01), deterministic hash ranking, two cases per bin and at most one case per original session per "
      "contrast. Shortfalls are disclosed, not filled by relaxed thresholds. All seed plans are shown; "
      "logged future ego trajectories are evaluation-only and overlays are not official simulated rollouts.")
    heading("8. Limitations, current status and stopping rule")
    p("The study concerns finite-width MLPs, selected privileged annotations and one short-horizon "
      "nonreactive protocol. It cannot establish regulatory-cue redundancy, end-to-end sensor performance, "
      "full perception redundancy, continuous replanning or deployment safety. Triggered secondary "
      "analyses do not become an unconditional confirmatory family. If applicable experiments cannot "
      "separate the competing explanations, the endpoint is a reproducible bounded technical report, "
      "not further architecture searches for a favorable story.")
    p("Complete controlled caches, three-split official scores/reports, actual follow-up gates, applicable "
      "extensions, scientific cases/figure QA and final end-to-end reproduction are still pending. The "
      "original controllers wait for safe resources and are not duplicated. This manuscript has no "
      "controlled PDMS results section or empirical mechanism conclusion. Final evidence and citation "
      "audits must precede any completed-paper claim; no upload or submission is authorized.", "warning")

    new_page()
    heading("References (version-bound; no baseline reproduction)")
    keys = ["dauner2023pdm", "dauner2024navsim", "zaheer2017deepsets", "cheng2023plantf", "li2024egostatus",
            "loewens2026memory", "cao2025pseudosim", "lee2019settransformer", "wang2026crossbenchmark", "zhang2026trafficelements"]
    authors = ["Dauner et al.", "Dauner et al.", "Zaheer et al.", "Cheng et al.", "Li et al.",
               "Loewens, Funke and Condurache", "Cao et al.", "Lee et al.", "Wang et al.", "Zhang et al."]
    labels = ["CoRL 2023; arXiv v2", "NeurIPS 2024 Datasets and Benchmarks; arXiv v2",
              "NeurIPS 2017; selected arXiv v3 sections", "arXiv v1, 2023; published ICRA 2024 title uses Planners",
              "CVPR 2024; arXiv v2", "arXiv v1 preprint, 2026", "CoRL 2025 / PMLR 305; arXiv v3 main review",
              "ICML 2019 / PMLR 97; publisher main and supplement", "arXiv v1 preprint, 2026",
              "arXiv v1, 2026; ECCV acceptance is author-reported"]
    indexed = {row["key"]: row for row in facts["metadata"]["references"]}
    for number, (key, author, label) in enumerate(zip(keys, authors, labels), 1):
        ref = indexed[key]
        url = escape(ref["bibliography_url"])
        p(f"[{number}] {escape(author)} ({ref['year']}). <i>{escape(ref['title'])}</i>. "
          f"{escape(label)}.<br/><link href='{url}' color='#245C85'>{url}</link>", "small")
    heading("Local evidence and reproducibility record")
    p("The companion render receipt binds the current protocol/donors, all checkpoint/history hashes, "
      "training table, bounded audit reports, source figure and bibliography review inputs. It records "
      "working-draft-only status and pending PDF visual QA. The original scientific receipts and source "
      "snapshots are preserved; neither a source ZIP nor this PDF establishes experimental reproduction.", "small")
    p("Authoritative local evidence: results/audit_v2 (core, cohort, live parity, process/cache recovery "
      "and city-input audits); results/mechanism_v2/protocol (base policies); training_diagnostics/summary.json; "
      "figures/training (source convergence and QA); docs/REPRODUCIBILITY.md (current command guide); "
      "docs/COMPLETION_REQUIREMENTS.md (full-scope ledger). Original protocols contain host-specific paths "
      "and must not be falsely rebound after relocation. Dataset/model-derived redistribution remains "
      "outside a source-code license grant.", "small")

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B6C1CA"))
        canvas.line(42, A4[1]-29, A4[0]-42, A4[1]-29)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#9D3027"))
        canvas.drawString(42, A4[1]-22, "WORKING DRAFT / CONTROLLED DRIVING RESULTS PENDING")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536472"))
        canvas.drawString(42, 24, "NAVSIM controlled study | 9 October 2026 | not a completed paper")
        canvas.drawRightString(A4[0]-42, 24, str(doc.page))
        canvas.restoreState()

    with output.open("xb") as stream:
        doc = SimpleDocTemplate(stream, pagesize=A4, leftMargin=42, rightMargin=42,
                                topMargin=43, bottomMargin=43, title="Current-object information: working NAVSIM manuscript",
                                author="Wenjie Luo", subject="Working draft; controlled scores pending")
        doc.build(story, onFirstPage=page, onLaterPages=page)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if any(path.is_symlink() for path in (args.out, *args.out.parents)):
        raise ValueError("Symlink destination refused")
    output = args.out.resolve()
    receipt_path = output.with_suffix(".receipt.json")
    if (not output.is_relative_to(ROOT / "output/pdf") or output.suffix != ".pdf"
        or output.exists() or receipt_path.exists()):
        raise ValueError("Use a fresh .pdf under output/pdf; preserve previous/failed attempts")
    facts = inspect_inputs(args.workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    render(output, facts)
    from pypdf import PdfReader
    pages = PdfReader(output).pages
    if not pages or any("WORKING DRAFT" not in (page.extract_text() or "") for page in pages):
        raise ValueError("Missing draft warning in the generated PDF; preserve attempt")
    report = {"status": "working_manuscript_rendered_visual_qa_pending", "created_utc": datetime.now(timezone.utc).isoformat(),
              "output": output.relative_to(ROOT).as_posix(), "pdf_sha256": digest(output), "pages": len(pages),
              "source_sha256": facts["source_sha256"], "protocol_sha256": facts["core"]["protocol_sha256"],
              "donors_sha256": facts["core"]["donors_sha256"], "checkpoint_sha256": facts["summary"]["checkpoint_sha256"],
              "history_sha256": facts["summary"]["history_sha256"], "n_verified_core_models": 18,
              "optimizer_steps_per_run": 13000, "scientific_binding_count": facts["scientific_binding_count"],
              "summary_recomputed_in_memory": True, "fitted_seed_mean_min_max_not_ci": True,
              "working_draft_only": True, "controlled_driving_scores_included": False,
              "scientific_manuscript_completed": False, "experimental_reproduction_completed": False,
              "publication_authorized": False, "final_project_completed": False,
              "heavy_modules_imported": [name for name in ("torch", "numpy", "matplotlib") if name in sys.modules],
              "scope": "Editable renderer and PDF working manuscript with current methods, selected related work, re-audited training diagnostics and bounded technical audits; no controlled PDMS results or final conclusions"}
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "output", "pdf_sha256", "pages", "heavy_modules_imported")}, indent=2))


if __name__ == "__main__":
    main()
