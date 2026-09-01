# NAVSIM ability ladder — what actually caps open-loop planning scores?

An ablation study on NAVSIM's `warmup_test_e2e` split. Instead of training one
planner and reporting a number, this builds a **ladder of ten agents where each
rung adds exactly one piece of information**, scores them all on the same metric
cache, and attributes the score gaps.

Every number below comes from `results/pdm_summary.csv`, which is produced by
NAVSIM's own `run_pdm_score.py`. The analysis scripts only read those CSVs — they
never re-run an evaluation, so nothing here can drift from the raw output.

## The ladder (563 scenes, same metric cache)

| agent | what it sees | deployable | PDMS | DAC |
|---|---|---|---:|---:|
| ConstantVelocity | speed only | yes | 0.233 | 0.643 |
| Kinematic | v, a, driving command | yes | 0.580 | 0.737 |
| PrivBrake | GT boxes, no map | no | 0.602 | 0.766 |
| EgoStatusMLP | learned, blind | yes | 0.640 | 0.806 |
| **MapMLP** | **learned, sees centerline** | no | **0.548** | 0.798 |
| PrivMap | map centerline + IDM | no | 0.786 | 0.934 |
| PrivMapKin | map centerline + kinematic | no | 0.802 | 0.950 |
| PrivGTPathKin | logged path + kinematic | no | 0.833 | 0.973 |
| PrivMapGTSpd | map centerline + human speed | no | 0.866 | 0.950 |
| Human | logged future | no | 0.945 | 0.998 |

`Priv*` and `Human` consume ground truth or map privilege. They are **upper
bounds, not results** — none of them is a deployable planner.

## What the ladder says

**1. Most of ConstantVelocity's pain is kinematics, not model capacity.**
Adding acceleration and command yaw alone moves PDMS 0.233 → 0.580 and cuts the
NC failure rate from 39% to 12%. Fix the motion model before reaching for a net.

**2. Perception boxes do not close the gap; the map does.**
GT detection boxes plus a brake reach only 0.602 with DAC stuck at 0.766.
Swapping in the on-route lane centerline moves DAC 0.737 → 0.950.

**3. Giving a learned model the map is not the same as it using the map.**
`MapMLP` is trained on the *identical* centerline features the hand-written
follower consumes. On the clean held-out split it scores 0.527 against the hand
rule's 0.730 — a gap of **+0.203, 95% CI [+0.118, +0.291]** in favour of the hand
rule, while the learned model's own gain over the blind baseline (+0.051) has a
CI that spans zero. With 3000 training scenes the model overfits (train L1 0.20,
val L1 1.4) rather than learning to follow the lane. The information is in the
features; the bottleneck is sample efficiency and inductive bias.

**4. Open-loop imitation quality is not closed-loop safety.**
The blind MLP has a low open-loop L1 and still gets zeroed by DAC on 90 of 563
scenes. A multiplicative safety metric does not care how close your trajectory
looked.

See `results/lab_notes.md` for the full write-up and
`results/ci_report.md` / `results/clean_ladder.md` for the statistics.

## Contamination, and how it is handled

The `mini` dataset ships 64 logs and **62 of them are the `warmup_test_e2e`
logs**. Any model trained on `mini` therefore has seen 428 of the 563 evaluation
scenes. Retraining on "mini minus warmup" is not an option — two logs would
remain.

So rather than quietly reporting a contaminated number, every agent is
re-scored on the same held-out val logs (`results/clean_ladder.md`, n=135). The
eight non-learned agents have no training set, so this only changes what the two
learned rows mean — and it makes them directly comparable to the rest.

**The cost is stated honestly.** At n=135 several effects that separated on 563
scenes no longer do: GT boxes (+0.020), dropping IDM (+0.012), swapping in the
logged path (+0.039), and the speed-vs-geometry difference-in-differences
(+0.041) all have CIs spanning zero. They are reported as suggestive, not
established.

## Known open issue

`privileged_brake_mini` was run twice and produced different results
(PDMS 0.602 / DAC 0.766, then 0.593 / 0.785). Every report here uses the first
run for consistency. **The discrepancy is unexplained** and should be pinned down
before that row is quoted anywhere.

## Layout

```
agents/     custom NAVSIM agents (kinematic, privileged brake/centerline, MapMLP)
configs/    matching hydra configs
analysis/   analyze_ci.py, analyze_clean_ladder.py, analyze_pdm.py, train_map_mlp.py
scripts/    PowerShell runners for each evaluation
results/    generated reports, pdm_summary.csv, failure trajectories
```

## Reproducing

This repository holds only original work. It is **not** a runnable checkout: the
NAVSIM devkit, the nuplan devkit, the OpenScene logs and the maps are all
upstream and separately licensed, and the dataset alone is ~3.8 GB.

1. Install NAVSIM v1.1 and download `mini` logs + maps per the upstream instructions.
2. Copy `agents/*.py` into `navsim/navsim/agents/` and `configs/*.yaml` into
   `navsim/navsim/planning/script/config/common/agent/`.
3. Set `NAVSIM_WORKSPACE`, `NAVSIM_EXP_ROOT`, `NAVSIM_DEVKIT_ROOT`,
   `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`; optionally `NAVSIM_PYTHON`.
4. Build the metric cache, then run `scripts/run_*.ps1` for each agent.
5. `python analysis/analyze_ci.py` and `python analysis/analyze_clean_ladder.py`
   regenerate the reports from the per-scene CSVs.

To retrain the map-aware model: `python analysis/train_map_mlp.py --epochs 80`
(CPU is fine — it is a 3-layer MLP on 3000 samples).

## Scope

- Built on the official NAVSIM agents and the official PDM score. **The scoring
  formula was never modified.** Windows portability fixes (packaging scope, POSIX
  file locks, token path splitting, certificate store) touched neither the metric
  nor any upstream agent.
- TransFuser was **not** run: `sensor_blobs/mini` is empty here and a GTX 1650
  4 GB cannot train it. Camera and LiDAR agents are therefore absent.
- `warmup_test_e2e` is not the official `navtest` leaderboard. None of these
  numbers is a leaderboard submission.

## Licence

MIT for the code in this repository. The NAVSIM and nuplan devkits are
Apache-2.0 upstream; the OpenScene/nuPlan data is under its own licence and is
not redistributed here.
