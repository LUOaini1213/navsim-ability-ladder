# Failure taxonomy on navtest: results

Where do the planners of the ladder fail, and which kinds of scenes carry the study's two
headline effects (the map-centerline gain, and the PDMS loss when ground-truth boxes are
added to a map-aware MLP)? The taxonomy, failure definitions, statistics, hypotheses and
root-cause rules were fixed in [`docs/FAILURE_TAXONOMY.md`](../../docs/FAILURE_TAXONOMY.md)
and committed (`83a7638`) before any feature was extracted or any rate computed; later
changes are listed under *Deviations* there. Every number below is in one of the generated
files named next to it.

**Status labels.** *Confirmatory* = one of the four hypotheses specified before the analysis
(navtest had already been used in this project, so this is not fresh-data confirmation).
Everything else is *exploratory*. Learned-cell intervals condition on the three fitted seeds.

## Data (`features_manifest.json`, `train_test_city_mix.json`)

- 12,146 navtest scenes, 136 logs, 44 sessions; eight features from the raw logs, the city
  maps (`intersections` layer) and the study's frozen centerline block. Extraction: one
  process, one thread, 21.5 s.
- Checked against the frozen feature archive the models were trained and scored with: speed
  differs by at most 9.4e-07 m/s, the lead gap by 1.9e-06 m, the logged 4-s pose by 1.9e-06 m;
  0 command mismatches. Command index order checked without outcomes: of 2,501 `left`
  commands, 2,354 have a logged left heading change and 26 a right one; of 1,575 `right`,
  1,407 turn right and 10 left.
- Category sizes: city vegas 4,064 / boston 3,731 / pittsburgh 2,437 / singapore 1,914;
  speed stopped 848 / slow 5,149 / medium 5,181 / fast 968; lead none 4,364 / close 2,876 /
  mid 3,224 / far 1,682; curvature straight 7,606 / bend 1,640 / turn 2,900; junction in 4,147
  / near 4,669 / none 3,330; lane change yes 1,260; stop-and-go starting 1,209 / stopping 901
  / stationary 7. No `unknown` driving command occurs.
- Singapore's 1,914 scenes come from 13 logs but only **2 sessions**, so by the
  pre-registered floor it never gets an interval. Las Vegas is 65.5% of navtrain (67,683 of
  103,288 scenes) but 33.5% of navtest; Singapore is 11.5% of navtrain and 15.8% of navtest.

## 1. Where the planners fail (exploratory; `where_planners_fail.md`, `category_outcomes.csv`)

PDMS by category (`where_planners_fail.md`):

| category | Kinematic | PrivMapKin | MLP kin | MLP kin+map | MLP kin+map+boxes |
|---|---:|---:|---:|---:|---:|
| all scenes | 0.489 | 0.785 | 0.642 | 0.823 | 0.782 |
| speed slow (0.5-5 m/s) | 0.633 | 0.798 | 0.730 | 0.832 | 0.801 |
| speed fast (>= 10 m/s) | 0.231 | 0.909 | 0.415 | 0.834 | 0.735 |
| no lead object | 0.298 | 0.796 | 0.535 | 0.804 | 0.736 |
| close lead (< 10 m) | 0.726 | 0.802 | 0.793 | 0.844 | 0.835 |
| starting from a stop | 0.670 | 0.754 | 0.879 | 0.918 | 0.912 |
| Singapore (*2 sessions*) | 0.287 | 0.806 | 0.584 | 0.802 | 0.663 |

- **Blind planners fail at speed and in free flow.** The kinematic rule falls from 0.633
  (slow) to 0.231 (fast) and the blind MLP from 0.730 to 0.415; both do worst without a lead
  object (0.298, 0.535) and best behind a close one (0.726, 0.793), where the scene mostly
  asks them to slow down.
- **Map-aware planners do not degrade with speed** (PrivMapKin 0.909 when fast, MLP kin+map
  0.834). Their residual failures shift to the longitudinal problem: the rule speed law is
  weak when starting from a stop (0.754 against 0.918 for MLP kin+map).
- **Drivable-area compliance is the dominant failure overall**: DAC fails in 36.5% of scenes
  for the kinematic rule and 23.7% for the blind MLP, against 7.0% and 7.3% for the
  map-aware rule and MLP (`where_planners_fail.md`, all-scenes row).

## 2. Which scene types carry the headline effects (exploratory; `headline_effects.md`)

κ = contribution of a category to the overall effect (categories of one feature sum to it);
*share* = κ / overall; seed averages unless stated.

**Map gain, rule (E1, +0.296) and learned (E2, +0.181).** Not a turn effect. It is carried by
free-flowing, moving traffic:

- no lead object: E1 +0.498 per scene (60% of the effect from 4,364 of the 12,146 scenes),
  E2 +0.269 (53%);
- fast scenes: E1 +0.678, E2 +0.419 (2.31 times the overall effect per scene);
- `moving` scenes carry 98% of E2; while starting (+0.039) or stopping (+0.005, interval
  spanning 0) the map adds almost nothing;
- by curvature E2 is +0.175 straight, +0.233 bend, +0.168 turn.

It works through drivable-area compliance: E2 cuts the DAC failure rate by 16.3 percentage
points overall, more than NC (5.3) or TTC (7.9) (`category_effects.csv`, `all` rows).

**Boxes added to the map-aware MLP (E3, -0.041; s0 -0.044, s1 -0.031, s2 -0.047).**

- Singapore: -0.139 per scene, **54% of the whole effect from 15.8% of the scenes**
  (seed shares 65% / 78% / 27%). Two sessions only, so no interval.
- No lead object: -0.067 [-0.100, -0.034] (log), 59% of the effect; close lead -0.009
  (interval spanning 0).
- Fast scenes -0.099, slow -0.030.
- Mechanism: more drivable-area failures (+3.3 points) and TTC failures (+2.3), not slower
  plans (EP<0.5 changes by -0.2 points) (`category_effects.csv`, `E3_avg`). In the root-cause
  sample the boxes model's failing plans differ from the map-only plans mostly sideways
  (10 of 24) rather than in speed (4 faster, 2 slower) (`cases/root_causes.md`).

**Map x boxes interaction (E4, -0.058).** Concentrated in fast scenes (-0.234 per scene,
4.03 times the average), straight commands (87% of the effect) and scenes away from
junctions (-0.093).

**City sensitivity (post hoc, exploratory; `city_sensitivity.md`).** Without Singapore the box
penalty shrinks to -0.022 [-0.036, -0.010] (log) / [-0.034, -0.008] (session) and is no longer
negative in every seed: -0.018 / -0.008 / -0.040, with seed 1's interval spanning 0 under
both clusterings and seed 0's under log clustering. The interaction E4 (-0.059 without
Singapore, every seed excluding 0) and the seed-average map gain (at least +0.163 with any
city left out)
are robust to dropping a city. So "boxes hurt in all three seeds" leans on two Singapore
sessions; "boxes hurt more once the map is there" does not.

## 3. Pre-registered hypotheses (confirmatory; `hypotheses.md`)

| hypothesis | seed average, log / session 95% | verdict |
|---|---|---|
| H1 learned map gain larger on curved routes | +0.017 [-0.014, +0.048] / [-0.015, +0.047] | not supported (seed 0 alone) |
| H1r rule map gain larger on curved routes | -0.033 [-0.080, +0.013] / [-0.083, +0.017] | not supported |
| H3a box penalty present without a lead | -0.067 [-0.100, -0.034] / [-0.102, -0.020] | not supported: holds for the average and seeds 1-2; seed 0 spans 0 under session clustering |
| H3b box penalty larger with a lead | +0.041 [+0.010, +0.073] / [+0.001, +0.069] | not supported: the sign is the opposite |

The penalty is *smaller* where a lead object is present, so it is not explained by the model
over-reacting to the lead input. Together with the mechanism above (sideways DAC failures,
mostly in free flow and in Singapore) this points at the 88 box dimensions degrading path
geometry rather than at the boxes being used for speed; that reading is exploratory.

## 4. Where clustering changes a conclusion (`clustering.md`, `hypotheses.md`)

- 11 of 104 category-level effect intervals change verdict between the iid scene bootstrap
  of the published tables and the log/session cluster bootstraps; cluster intervals are a
  median 2.07 (log) and 2.13 (session) times as wide. In 9 of the 11 the iid interval
  excludes 0 and both cluster intervals span it (for example E3 in Pittsburgh, -0.020: iid
  [-0.032, -0.008], log [-0.071, +0.012]); in E3 bends and E1 lane changes only session
  clustering spans 0.
- H3a seed 0 is decided by the clustering unit: log [-0.111, -0.007], session [-0.122, +0.019].
- Overall, E3 seed 1 spans 0 under session clustering (-0.031, [-0.064, +0.000];
  `headline_effects.md`), while the published iid interval excludes it.

## 5. Root causes on a stratified sample (`cases/root_causes.md`, `cases/cases.csv`)

120 failures (PDMS < 0.5), 24 per stratum, at most one per log, hash-ordered: S1 kinematic
failures fixed by PrivMapKin, S2 blind-MLP failures fixed by the map MLP, S3 failures the
boxes introduce into the map MLP, S4 residual map-MLP failures, S5 residual PrivMapKin
failures. Every plan was rebuilt and re-scored with the unchanged official scorer:
**120 of 120 failing plans and 72 of 72 contrast plans reproduce** the committed per-scene
rows (largest error 5e-07).

- DAC is the primary failure in 78 of 120. The most common label in every stratum is
  `dac_near_logged_path`, an off-road excursion while within 1 m of the logged path: 14/24
  (S1), 9/24 (S2), 11/24 (S3), 10/24 (S4), 6/24 (S5).
- The excursions are small: median distance to the logged path at the first off-road step
  0.63 m; over the horizon the worst ego corner gets a median 0.62 m outside the drivable
  polygons (0.35 m for S3), and in 13 of 78 it never gets 0.1 m outside. None of the 78
  off-road corners lies on a mapped lane that the drivable polygons miss, so lane-coverage
  gaps do not explain them. A multiplicative DAC turns a few decimetres into a zero.
- Rear-end collisions are the second mode, largest among the map MLP's residual failures
  (9/24, 38%, Wilson 21-57%).
- Gallery: 12 bird's-eye views in `cases/gallery/` (map polygons, on-route lanes, route
  centerline, agents at t0 and at the failure step, plan vs logged trajectory, contrast plan,
  ego footprint at the first failure), listed with captions in `cases/root_causes.md`.

## Caveats

- navtest had informed earlier hypotheses; seeds are three fitted models, not a population.
- Singapore rests on two sessions; the city findings are the least certain ones here.
- Features overlap (no lead, fast and free flow go together; Singapore has its own mix).
  Per-feature breakdowns are marginal, not adjusted for each other.
- `lane_change` and `stop_go` use the logged future: they describe what a scene asked for,
  not what a planner could see.
- Root-cause labels are deterministic rules over the scorer's own state; 24 cases per
  stratum gives wide Wilson intervals. Learned-cell plans come from the frozen inputs; every
  one was checked by re-scoring.
- NAVSIM v1 PDMS is a non-reactive simulation score, not closed-loop driving.

## Reproduce

```bash
python -m unittest tests.test_failure_taxonomy -v       # core rules, bootstrap replay, committed point estimates
python analysis/failure_taxonomy_analyze.py --check     # every committed mean/contribution from the CSVs (stdlib)
python analysis/failure_taxonomy_analyze.py --out FRESH  # all tables and intervals (numpy: ~2.5 min)
# with the NAVSIM workspace (raw logs, maps, metric cache, checkpoints), into fresh folders:
python analysis/failure_taxonomy_features.py --out FRESH --threads 1
python analysis/failure_taxonomy_cases.py select --out FRESH_CASES
python analysis/failure_taxonomy_cases.py score --source archive --out FRESH_CASES
python analysis/failure_taxonomy_cases.py score --source scene --city vegas --out FRESH_CASES   # and the other cities
python analysis/failure_taxonomy_cases.py report --out FRESH_CASES
```
