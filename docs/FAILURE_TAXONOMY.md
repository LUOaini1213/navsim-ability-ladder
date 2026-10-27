# Failure-case taxonomy on navtest: pre-registration

This file fixes the scene taxonomy, the failure definitions, the statistics, the
hypotheses and the root-cause labelling rules **before** any of them is computed.
It is committed on its own, ahead of any feature extraction or analysis code, so
the order is visible in the history. Results go to `results/failure_taxonomy/`;
anything changed after this commit is listed under *Deviations* at the end, with
the commit that made the change.

## What was known when this was written

- The published navtest tables (`results/navtest/*.md`, README): overall PDMS,
  sub-score means and overall NC/DAC/TTC failure rates for every run, and the
  overall paired effects (map gain, box effects, interaction).
- The overall log- and session-clustered intervals of the main contrasts
  (`results/navtest/clustered.md`, `session_clustered.md` on the open
  controlled-study branch).
- No scene feature had been extracted and no failure rate, sub-score or effect
  had been computed for any subset of scenes. The feature distributions had not
  been looked at either: the bins below are fixed from driving-domain
  conventions, not from the data.

navtest has already been used in this project to develop hypotheses. Nothing
here is fresh-data confirmation; "confirmatory" below means "specified in this
file before the analysis", nothing more.

## Questions

1. In which kinds of scenes do the planners fail?
2. Which scene types carry the study's headline effects: the large PDMS gain
   from adding the map centerline, and the PDMS loss from adding ground-truth
   boxes to a map-aware MLP (negative in all three seeds while open-loop L1
   barely moves)?

## Unit of analysis

One navtest scene = one token of `navtest.yaml`: 12,146 scenes from 136 logs.
A *session* is the log name without its last two `_`-separated fields
(`log.rsplit('_', 2)[0]`, vehicle + start time): 44 sessions. `t0` is the
scene's current frame (index 3 of the 14-frame window: 4 history frames, 10
future frames at 2 Hz, `frame_interval = 1`). The scored horizon is 4 s (the 8
future frames after `t0`).

## Scene features and fixed bins

Every feature is computed for every scene. A scene whose input for a feature is
missing gets the category `n/a` for that feature (reported, never imputed).

| feature | source | categories (fixed) |
|---|---|---|
| `city` | `map_location` of the `t0` log frame | `vegas` (us-nv-las-vegas-strip), `pittsburgh` (us-pa-pittsburgh-hazelwood), `boston` (us-ma-boston), `singapore` (sg-one-north) |
| `speed` | ego speed `v0 = hypot(vx, vy)` from `ego_dynamic_state` at `t0`, m/s | `stopped` v0 < 0.5; `slow` 0.5 ≤ v0 < 5; `medium` 5 ≤ v0 < 10; `fast` v0 ≥ 10 |
| `command` | argmax of `driving_command` at `t0` | `left` (0), `straight` (1), `right` (2), `unknown` (3) |
| `lead` | closest dynamic object in the ego corridor at `t0`, from the log's GT boxes | `none`; `close` gap < 10 m; `mid` 10 ≤ gap < 25 m; `far` gap ≥ 25 m |
| `junction` | map `intersections` polygons + on-route centerline ahead | `in` (ego inside an intersection); `near` (an intersection within the first 30 m of route ahead); `none` |
| `curvature` | heading change of the on-route centerline between 0 and 30 m ahead | `straight` < 15°; `bend` 15°–45°; `turn` ≥ 45° |
| `lane_change` | lateral displacement of the logged ego path relative to the route centerline over 4 s | `no` (abs Δd < 2.0 m); `yes` (abs Δd ≥ 2.0 m) |
| `stop_go` | logged ego speeds at `t0` and the 8 future frames | `stationary`; `starting`; `stopping`; `moving` |

Definitions, exactly as they will be implemented:

- **command.** The index order left / straight / right / unknown is the one
  `agents/kinematic_agent.py` assumes. As a sanity check that uses no outcome,
  the sign of the logged 4-s heading change is tabulated against the command.
- **lead.** The rule of `PrivilegedBrakeAgent._lead_gap_speed`
  (`agents/privileged_brake_agent.py`), which is also the rule that produces the
  boxes models' `lead` input: object class in {vehicle, pedestrian, bicycle,
  generic_object}; box centre (ego frame) with 1.5 m < x < 55 m and
  abs(y) ≤ 2.3 + 0.25·width (at least 3.0 m for pedestrians); bumper gap
  `x − length/2 − 4.0`. The closest object by gap is the lead. Negative gaps
  fall in `close`. The lead's class is stored for description. The extracted gap
  is checked against the frozen `lead` block of the feature archive.
- **route centerline.** The study's frozen `cl` block
  (`exp/features/features_navtest.npz`): 20 samples at 0, 60/19, …, 60 m along
  the on-route lane centerline in the `t0` ego frame, built from the log's
  `roadblock_ids` and the map's lane baselines by `agents/centerline_util.py`.
  It is the centerline the map-aware models were given.
- **junction.** Intersection polygons are read from the city's `map.gpkg`
  (layer `intersections`) and projected to the map's UTM system exactly as
  nuplan's `GPKGMapsDB.load_vector_layer` does. `in`: the ego position at `t0`
  (log `ego2global_translation`) lies inside a polygon. Otherwise `near`: at least
  one centerline sample with arc length ≤ 30 m, transformed to global with the
  `t0` ego pose, lies inside a polygon. Otherwise `none`. The arc length of the
  first inside sample (0 for `in`, empty if none within 60 m) is kept as
  `junction_dist_m` for exploratory use only.
- **curvature.** The centerline heading (from the `cl` cos/sin columns) is
  unwrapped along the 20 samples and linearly interpolated at 30 m; the feature
  is abs(heading(30 m) − heading(0 m)).
- **lane_change.** Signed lateral offset `d` of a point from the `cl` polyline
  (the last segment extended in a straight line beyond 60 m). `Δd = d(p8) −
  d(p0)`, where p0 is the ego position at `t0` (the origin) and p8 the logged ego
  position 4 s later (log `ego2global` of frame `t0+8`, in the `t0` ego frame).
  This flags a lane change *or* a departure from the route lane the centerline
  follows; it uses the logged future, so it describes what the scene asked for
  and is not information a planner had.
- **stop_go.** With `v0` the speed at `t0` and `v1..v8` the logged speeds of the
  8 future frames: `stationary` if max(v0..v8) < 0.5; else `starting` if v0 < 1.0
  and max(v1..v8) ≥ 2.0; else `stopping` if v0 ≥ 2.0 and min(v1..v8) < 0.5; else
  `moving`. "Stop-and-go" = `starting` ∪ `stopping`. Uses the logged future, like
  `lane_change`.

The extracted features are cached once, in a fresh `results/failure_taxonomy/`,
with a manifest of inputs, counts and consistency checks against the frozen
feature archive. Extraction runs serially (one process, at most 2 threads).

## Outcomes

Per scene and run, from `results/navtest/per_scene/*.csv.gz` (the committed
output of NAVSIM's `run_pdm_score.py`; nothing is re-scored for the tables):

| name | definition |
|---|---|
| `pdms` | the scene score (mean reported) |
| `fail_pdms` | score < 0.5 |
| `fail_nc` | no-at-fault-collision < 1 (0 = collision with an agent, 0.5 = with a static object) |
| `fail_dac` | drivable-area compliance < 1 |
| `fail_ttc` | time-to-collision-within-bound < 1 |
| `fail_c` | comfort < 1 |
| `fail_ep` | ego progress < 0.5 **and** NC·DAC > 0 (slow plan, not a by-product of a multiplicative zero, which forces EP to 0) |

Driving-direction compliance has weight 0 in the v1 score and is not used.

## Runs and headline effects

Runs: every rule / official run (`cv`, `kin`, `privbrake`, `egomlp_s0..2`,
`privmap_idm`, `privmapkin`, `privgtpath_kin`, `privmap_gtspd`, `human`) and the
learned cells `l_pose_{kin,agents,map,map_agents}_s{0,1,2}` and
`l_ds_{kin,map,map_agents}_s{0,1,2}`. A learned cell's seed average is the
per-scene mean over its three seeds, as in `analysis/analyze_clustered.py`.

| id | effect | contrast (per scene) |
|---|---|---|
| E1 | rule map gain | `privmapkin − kin` |
| E2 | learned map gain (no boxes) | `l_pose_map_s − l_pose_kin_s` |
| E3 | boxes added to the map-aware MLP | `l_pose_map_agents_s − l_pose_map_s` |
| E4 | map × boxes interaction | `(l_pose_map_agents_s − l_pose_map_s) − (l_pose_agents_s − l_pose_kin_s)` |

E2–E4 are reported per seed (s = 0, 1, 2) and for the seed average.

## Statistics

- **Category means** (PDMS, failure rates, effects): the scene-weighted mean of
  the category's scenes, with the paired percentile bootstrap of
  `analysis/analyze_clustered.py` (`cluster_ci`, open controlled-study branch,
  commit `ad049d1`): whole clusters resampled with replacement, ratio of
  resampled sums to resampled scene counts, 10,000 draws, seed 12345, 2.5th and
  97.5th percentiles. Clusters are logs (136) and, separately, sessions (44).
  Applied to the category's own scenes, exactly as that function is written.
- **Contributions.** For effect Δ and category c, κ_c = Σ_{t∈c} Δ_t / N, so the
  categories of one feature sum to the overall effect. Interval: `cluster_ci`
  on all N scenes with values Δ_t·1[t ∈ c]. Share = κ_c / Δ; concentration =
  share / (n_c / N) = Δ_c / Δ (above 1: the category carries more than its scene
  share).
- **Differences between groups of categories** (the confirmatory tests): D =
  Δ_A − Δ_B. All clusters are resampled with the same random stream as
  `cluster_ci`; both group means are recomputed in every draw; a draw in which a
  group is empty is discarded and the number of discarded draws is reported.
- A vectorised implementation that replays `cluster_ci`'s random stream is used
  for speed; unit tests require it to reproduce `cluster_ci` to 1e-12.
- **iid reference.** The scene-level bootstrap used by the published tables
  (`bootstrap_large` in `analysis/ladder_io.py`, 2,000 draws, seed 12345) is
  computed for the effect tables only, to show where clustering changes a
  verdict.
- **Verdict** of an interval: "excludes 0" or "spans 0". A conclusion *changes
  with clustering* when the verdicts under iid, log and session clustering are
  not all the same; every such case is listed.
- **Reporting floor.** Intervals and verdicts only for categories with at least
  50 scenes, 5 logs and 3 sessions; smaller categories get a point estimate
  marked *small*.
- Learned-cell intervals condition on the three fitted models; they do not
  estimate training-seed uncertainty. Exploratory tables carry no multiplicity
  correction.

## Confirmatory hypotheses

Each is decided under **both** log and session clustering; for E2/E3 it must
hold for the seed average **and** for each of the three seeds. Anything short of
that is reported as "not supported" with the per-seed detail.

- **H1** (the learned map gain is larger on curved routes). D = Δ_E2(curvature ∈
  {bend, turn}) − Δ_E2(curvature = straight) > 0, interval excluding 0.
- **H1r** (same for the rule map gain). D = Δ_E1(bend ∪ turn) − Δ_E1(straight)
  > 0, interval excluding 0 (single run: log and session only).
- **H3a** (the box penalty is present without a lead). Δ_E3(lead = none) < 0,
  interval excluding 0. Support means the penalty is not explained by the lead
  input alone.
- **H3b** (the box penalty is larger with a lead). D = Δ_E3(lead ∈ {close, mid,
  far}) − Δ_E3(lead = none) < 0, interval excluding 0.

H3a and H3b are not exclusive. Everything else in the analysis is exploratory.

**Expectations** (written down for transparency, not tested): the blind
planners' (`kin`, `l_pose_kin`) DAC failures concentrate in `curvature = turn`
and left/right commands; the map-aware planners' remaining failures are mostly
NC/TTC with close leads and in stop-and-go; the boxes models' extra failures sit
in `lead = close` and `starting` scenes.

## Root-cause labelling (rules fixed before any case is labelled)

**Strata.** A failure is PDMS < 0.5.

| stratum | condition | failing run that is labelled |
|---|---|---|
| S1 rule map-fixed | `kin` < 0.5 and `privmapkin` ≥ 0.5 | `kin` |
| S2 learned map-fixed | `l_pose_kin_s` < 0.5 and `l_pose_map_s` ≥ 0.5 | `l_pose_kin_s` |
| S3 box-induced | `l_pose_map_s` ≥ 0.5 and `l_pose_map_agents_s` < 0.5 | `l_pose_map_agents_s` |
| S4 learned residual | `l_pose_map_s` < 0.5 | `l_pose_map_s` |
| S5 rule residual | `privmapkin` < 0.5 | `privmapkin` |

**Sample.** 24 cases per stratum (S2–S4: 8 per seed). Candidates are ordered by
`sha256("p10-root-cause|<stratum>|<seed>|<token>")`; at most one case per log per
stratum (a case from an already used log is skipped); fewer if the stratum runs
out. Shares are reported with Wilson 95% intervals and are descriptive.

**Re-scoring.** Each case is rebuilt with NAVSIM's `SceneLoader` from the raw
log, the unchanged agent is run (the devkit copy, identical to `agents/` here
apart from line endings; learned cells load their archived checkpoint), and the
plan is scored with the unchanged official scorer against the navtest metric
cache. The case counts as reproduced if PDMS and all six sub-scores are within
1e-4 of the committed per-scene row; otherwise it is labelled `unreproduced`
and left out of the shares.

**Primary failure.** The earlier of the NC event (first simulation step with an
at-fault collision) and the DAC event (first step with an ego corner outside the
drivable area), ties going to DAC; if neither, TTC if TTC = 0; else comfort if
C = 0; else `other`.

**Labels**, applied in this order:

1. `fallback` — a learned agent did not use its network (unusable route or
   exception, so the kinematic fallback produced the plan).
2. DAC primary. Lateral distance = distance from the plan's simulated position
   at the DAC step to the logged 4-s path (polyline from the origin through the
   8 logged poses, extended straight along the last heading).
   - `dac_lateral_curve`: lateral distance ≥ 1.0 m and abs(logged 4-s heading
     change) ≥ 15°.
   - `dac_lateral_straight`: lateral distance ≥ 1.0 m and heading change < 15°.
   - `dac_overshoot`: lateral distance < 1.0 m and the plan's travelled distance
     at the DAC step exceeds the logged 4-s path length by ≥ 1.0 m.
   - `dac_near_logged_path`: otherwise.
3. NC primary (the at-fault object is found by repeating the scorer's own query
   and collision classification at the NC step).
   - `nc_static`: NC = 0.5.
   - `nc_vru`: the object is a pedestrian or bicycle.
   - `nc_lateral`: the collision is classified as an active lateral collision.
   - `nc_rear_end`: heading difference to the object < 30° and the object's
     centre ahead of the ego (x > 0 in the ego frame at that step).
   - `nc_crossing`: otherwise.
4. TTC primary: `ttc_following` (the object found by the scorer's TTC query is
   ahead with heading difference < 30°), else `ttc_crossing`.
5. `comfort`.
6. `other`.

For S3 a second descriptor compares the boxes model's plan with the map-only
model's plan of the same seed: `slower` (4-s travelled distance at least 2 m
shorter), `faster` (at least 2 m longer), `lateral` (endpoints at least 0.5 m
apart laterally, if neither), else `similar`.

**Gallery.** About 12 bird's-eye-view plots: map polygons and route lanes from
the metric cache, the route centerline, agents at `t0` and at the failure step,
the plan, the logged trajectory and, for S1–S3, the contrasting run's plan.
Selection: per stratum the two reproduced cases with the stratum's most common
label, then one case of the second most common label in S2 and in S3, each pick
in hash order — 12 in total, fewer if a stratum is short.

## Deviations

Everything below was added after the pre-registration commit (`83a7638`). Results:
`results/failure_taxonomy/README.md`.

1. **`lane_change`, routes shorter than 60 m** (found in a 3-log smoke test, before any outcome
   was read). The frozen `cl` block stores a route that ends before 60 m with its end point
   repeated. Repeated points are now dropped before the last real segment is extended;
   otherwise the "extension" is a zero-length segment and the offset is measured to a point.
   Twelve scenes have a zero-length route and get `lane_change = n/a`.
2. **Extra raw column `route_len_m`** (length of the stored centerline, which is shorter than
   the 60-m lookahead when the route ends). Kept for exploratory use; it is not a category.
3. **Where the case plans come from.** The text above says every case is rebuilt with
   `SceneLoader`. That is done for PrivMapKin (S1 contrast, S5). The kinematic rule and the
   learned cells (S1–S4) are rebuilt from the study's frozen inputs instead: `KinematicAgent`
   fed the `t0` ego status it reads, and the network branch of `LadderMLPAgent` on the archived
   input blocks (the score path of `analysis/validate_score_path.py` on the controlled-study
   branch). This keeps the map out of memory on a shared machine. The acceptance rule was
   applied unchanged: all 192 plans (120 failing, 72 contrast) reproduce the committed
   per-scene rows (largest absolute sub-score error 4.8e-07 for the failing plans, 5e-07 for
   the contrast plans). With this source `fallback` can only mean a non-finite network
   output; it never occurred.
4. **Post-hoc, exploratory additions** (none changes a category, a label or a hypothesis):
   - `city_sensitivity.md`: every headline effect with one city left out, added after the city
     breakdown showed Singapore carrying a large share of the box penalty;
   - descriptive drivable-area margins in `cases/root_causes.md` (distance of the plan from the
     logged path at the first off-road step, how far the worst ego corner gets outside the
     DAC polygons, whether that corner lies on a mapped lane), added after the first look at
     the labels showed most DAC failures within 1 m of the logged path.
5. **iid reference, narrower than written.** The scene-level bootstrap was computed for E1 and
   the seed-averaged E2–E4 only, not for every seed, to keep the run short; the per-seed rows
   carry log and session intervals only.
6. **Note, not a change.** By the reporting floor, Singapore (1,914 scenes, 13 logs, but 2
   sessions) gets no interval anywhere; every Singapore statement rests on two sessions.
