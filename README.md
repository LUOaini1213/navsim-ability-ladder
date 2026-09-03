# NAVSIM ability ladder — what actually caps open-loop planning scores?

[![ci](https://github.com/LUOaini1213/navsim-ability-ladder/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOaini1213/navsim-ability-ladder/actions/workflows/ci.yml)

An ablation study on NAVSIM's `warmup_test_e2e` split. Instead of training one
planner and reporting a number, this builds a **ladder of agents where each rung
adds exactly one piece of information**, scores them all on the same metric
cache, attributes the score gaps — and then tries three ways of putting a
learned component back in, to see which slot it can actually fill.

Every number below comes from per-scene CSVs written by NAVSIM's own
`run_pdm_score.py`. The analysis scripts only read those CSVs — they never
re-run an evaluation, so nothing here can drift from the raw output. **The CSVs
are in this repository** (`results/per_scene/`, 13 agents × 563 scenes, 616 KB),
so every table and confidence interval reproduces from a plain clone:

```bash
python -m unittest discover -s tests -v          # regenerates results/*.md from the CSVs and compares byte for byte
python analysis/analyze_ci.py                    # -> results/ci_report.md      (10000 paired resamples, seed 12345)
python analysis/analyze_clean_ladder.py          # -> results/clean_ladder.md   (the 135 held-out scenes)
python analysis/make_summary.py --check README.md   # the numbers in the tables below, recomputed
```

Thirteen agents were scored in total: the nine rungs of the ladder below, the
learned `MapMLP`, and its three variants (`MapMLP-reg`, `SpeedMLP`,
`SpeedMLP` at 200 epochs). Tables in `results/` list all thirteen; the README
tables show the rungs plus the variants that changed a conclusion.

## The ladder (563 scenes, same metric cache)

| agent | what it sees | deployable | PDMS | DAC |
|---|---|---|---:|---:|
| ConstantVelocity | speed only | yes | 0.233 | 0.643 |
| Kinematic | v, a, driving command | yes | 0.580 | 0.737 |
| PrivBrake | GT boxes, no map | no | 0.602 | 0.766 |
| EgoStatusMLP | learned, blind | yes | 0.640 | 0.806 |
| MapMLP | learned, sees centerline | no | 0.548 | 0.798 |
| PrivMap | map centerline + IDM | no | 0.786 | 0.934 |
| PrivMapKin | map centerline + kinematic | no | 0.802 | 0.950 |
| **SpeedMLP** | **map centerline + learned speed** | no | **0.806** | **0.964** |
| PrivGTPathKin | logged path + kinematic | no | 0.833 | 0.973 |
| PrivMapGTSpd | map centerline + human speed | no | 0.866 | 0.950 |
| Human | logged future | no | 0.945 | 0.998 |

`Priv*` and `Human` consume ground truth or map privilege. They are **upper
bounds, not results** — none of them is a deployable planner. Learned rows on
this split are contaminated (see below); their clean numbers are in the next
table.

## What the ladder says

**1. Most of ConstantVelocity's pain is kinematics, not model capacity.**
Adding acceleration and command yaw alone moves PDMS 0.233 → 0.580 and cuts the
NC failure rate from 39% to 12%. Fix the motion model before reaching for a net.

**2. Perception boxes do not close the gap; the map does.**
GT detection boxes plus a brake reach only 0.602 with DAC stuck at 0.766.
Swapping in the on-route lane centerline moves DAC 0.737 → 0.950.

**3. Giving a learned model the map is not the same as it using the map.**
`MapMLP` is trained on the *identical* centerline features the hand-written
follower consumes, and asked to regress the whole trajectory. On the clean split
it scores 0.527 against the hand rule's 0.730 — **+0.203, 95% CI [+0.118,
+0.291]** in favour of the hand rule. The information is in the features; the
bottleneck is sample efficiency and inductive bias.

**4. Open-loop imitation quality is not closed-loop safety.**
The blind MLP has a low open-loop L1 and still gets zeroed by DAC on 90 of 563
scenes. A multiplicative safety metric does not care how close your trajectory
looked.

![One of the 90 DAC failures: progress vs time (left) and the local x-y plan (right) for the human trajectory (black), ConstantVelocity (blue, dashed) and the blind EgoStatusMLP (red). The MLP matches the human progress almost exactly but under-turns laterally; that path leaves the drivable area and DAC multiplies the scene score to zero.](results/figures/traj_warmup_fail_DAC_dace7f508e4b5070.png)

*One of the 90: the blind MLP keeps up with the human in progress (left) and still loses the scene, because it under-turns by a few decimetres (right) and the drivable-area check is multiplicative. NC and TTC failure examples are in `results/figures/`.*

## Putting a learned part back in: three attempts

Three ways of fixing `MapMLP`, all trained on the same 51 train logs and scored
on the same 135 held-out scenes (`results/clean_ladder.md`).

| variant | what the model predicts | open-loop val L1 | PDMS, clean n=135 | vs hand rule PrivMapKin (0.730) |
|---|---|---:|---:|---|
| MapMLP | whole xyθ trajectory | 0.99 | 0.527 | −0.203 [−0.291, −0.118] |
| MapMLP-reg (h128, dropout 0.2, wd 1e-3) | whole xyθ trajectory | 0.65 | 0.425 | −0.305 [−0.396, −0.215] |
| **SpeedMLP** | **only progress along the hand-drawn centerline** | 0.76 | **0.763** | **+0.033 [−0.016, +0.083]** |
| SpeedMLP, 200 epochs | same | 0.65 | 0.747 | +0.017 |
| PrivMapGTSpd (human speed, upper bound) | — | — | 0.809 | +0.079 |

**5. The decomposition works; more capacity and more regularisation do not.**
Asked to draw the whole line, the learned model loses to the hand rule by 0.2.
Asked only to *pace* a line the rule has drawn, it matches the rule (0.763 vs
0.730, CI spans zero; 0.806 vs 0.802 on the full split) with fewer drivable-area
failures (3.6% vs 5.0%). A learned component earns its place in the slot where
the hand rule is weakest — the speed profile — not in the slot the rule already
handles.

**6. Lower open-loop loss made closed-loop worse, twice.**
Regularising `MapMLP` cut its open-loop val L1 from 0.99 to 0.65 and cut its
PDMS from 0.527 to 0.425 (−0.102, CI [−0.182, −0.024]). Training `SpeedMLP` for
200 epochs instead of 80 cut L1 from 0.76 to 0.65 and moved PDMS from 0.763 to
0.747 (CI spans zero). Selecting models by open-loop L1 would have picked the
wrong one both times. Finding 4, from the other direction.

**7. Learned speed recovers part of the gap, not all of it.**
Hand speed 0.730 → learned speed 0.763 → human speed 0.809 on the clean split.
The learned profile is still +0.047 [+0.005, +0.091] short of the human one; the
remaining headroom is in how fast to go, not where to go.

See `results/lab_notes.md` for the original write-up and
`results/ci_report.md` / `results/clean_ladder.md` for every CI.

## Contamination, and how it is handled

The `mini` dataset ships 64 logs and **62 of them are the `warmup_test_e2e`
logs**. Any model trained on `mini` therefore has seen 428 of the 563 evaluation
scenes. Retraining on "mini minus warmup" is not an option — two logs would
remain.

So rather than quietly reporting a contaminated number, every agent is
re-scored on the same held-out val logs (`results/clean_ladder.md`, n=135). The
non-learned agents have no training set, so this only changes what the learned
rows mean — and it makes them directly comparable to the rest.

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

## What is and is not in this repository

| In the repo | Not in the repo |
|---|---|
| the five agents and their hydra configs (`agents/`, `configs/`) | the NAVSIM / nuplan devkits, the OpenScene logs, the maps (~3.8 GB, upstream licences) |
| per-scene PDM scores of all 13 agents, the scene→log map and the train/val split (`results/per_scene/`) | the metric cache and the trained checkpoints (`*.ckpt`) |
| the analysis scripts, and tests that regenerate every report from the CSVs (`analysis/`, `tests/`) | the workspace runner `run_navsim_step.py` the PowerShell scripts call |
| the reports, the 13-agent summary and six failure-trajectory figures (`results/`) | the sensor blobs (empty here; no camera / LiDAR agent was run) |

## Layout

```
agents/     kinematic, privileged brake / centerline, MapMLP, SpeedMLP, centerline_util
configs/    matching hydra configs (paths via ${oc.env:NAVSIM_EXP_ROOT})
analysis/   ladder_io.py (shared loading + bootstrap), analyze_ci.py, analyze_clean_ladder.py,
            make_summary.py, analyze_pdm.py, train_map_mlp.py, train_speed_mlp.py
tests/      bootstrap sanity on synthetic data; reports regenerate byte for byte from results/per_scene
scripts/    PowerShell runners for the evaluations (need the workspace)
results/    per_scene/ CSVs, generated reports, pdm_summary.csv, failure trajectories
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
4. Build the metric cache, then run `scripts/run_*.ps1` for each hand-written agent.
5. Train the learned agents (CPU is fine — 3-layer MLPs on ~3000 samples):
   `python analysis/train_map_mlp.py --epochs 80`
   `python analysis/train_map_mlp.py --epochs 80 --hidden 128 --dropout 0.2 --wd 1e-3 --tag reg`
   `python analysis/train_speed_mlp.py --epochs 80 --hidden 128 --dropout 0.2 --wd 1e-3`
   then score them with `run_pdm_score.py agent=map_mlp_agent|map_mlp_reg_agent|speed_mlp_agent`.
6. `python analysis/analyze_ci.py` and `python analysis/analyze_clean_ladder.py`
   regenerate the reports from the per-scene CSVs — from the workspace when
   `NAVSIM_EXP_ROOT` points at it, otherwise from `results/per_scene/`, which is
   why steps 1–5 are not needed to check any number in this README.

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
