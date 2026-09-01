# NAVSIM lab notes (warmup_test_e2e, n=563)

Last updated: 2026-08-20

## Ladder (same metric cache, same split)

| agent | PDMS | vs CV | DAC | what it sees | deployable? |
|---|---:|---:|---:|---|---|
| ConstantVelocity | **0.233** | 0 | 0.643 | speed only | yes |
| Kinematic | **0.580** | +0.347 | 0.737 | v, a, command | yes |
| PrivBrake (GT boxes) | **0.602** | +0.369 | 0.766 | detections, no map | **no** |
| EgoStatusMLP | **0.640** | +0.407 | 0.806 | learned kinematics | yes |
| PrivMap (centerline+IDM) | **0.786** | +0.553 | 0.934 | map + route + IDM | **no** |
| PrivMapKin (centerline+kin) | **0.802** | +0.569 | 0.950 | map, no IDM | **no** |
| PrivGTPathKin (GT path+kin) | **0.833** | +0.600 | 0.973 | logged path, kin speed | **no** |
| PrivMapGTSpd (centerline+GT s) | **0.866** | +0.632 | 0.950 | map + Human arc-length | **no** |
| Human (GT path+GT time) | **0.945** | +0.711 | 0.998 | logged future | **no** |

563/563 scored, 0 failed jobs.

TransFuser was **not** run: `dataset/sensor_blobs/mini` is empty (0 GB). Official checkpoint would still need cameras+LiDAR. GTX 1650 4GB cannot train it.

## What this means

1. **Most of CV’s pain is kinematics, not a huge net.**  
   Accel + command yaw: PDMS 0.23 → 0.58. NC fail 39% → 12%.
2. **MLP is still blind** and only +0.06 over the hand kinematic rule.
3. **GT boxes + brake do not close the Human gap** (0.580 → 0.602). DAC stays ~0.77.
4. **On-route map centerline does.** DAC 0.73→0.95. IDM on the centerline *hurts* slightly (0.786 vs kin 0.802).
5. **Geometry vs speed 2×2** (same 563):

   | PDMS | kinematic speed | Human arc-length |
   |---|---:|---:|
   | map centerline | 0.802 | **0.866** |
   | logged GT path | 0.833 | **0.945** |

   Speed on a *good path* is the bigger remaining slice (centerline + GT s already 0.866). GT path + kin only 0.833 and comfort drops to 0.84 (resampling logged poses at the wrong timing). Combined, leftover Human is both: lane-center ≠ logged polyline, and v(t) ≠ human.

## MLP failure families

See `exp/analysis/mlp_fail_counts.json` and `exp/trajectories/traj_warmup_fail_*.png`.

- ok 359 / DAC 90 / TTC 63 / NC 51

## Commands

```powershell
.\run_kinematic_mini.ps1
.\run_human_mini.ps1
.\run_privileged_mini.ps1
.\run_centerline_mini.ps1
.\run_path_ablation.ps1
python analyze_pdm.py
```

## Online eval vs paper (navtest)

See `exp/analysis/online_eval.md`. Short version (scores in **percent**):

| | ours warmup | paper navtest |
|---|---:|---:|
| CV | 23.3 | 20.6 |
| EgoStatusMLP | 64.0 | 65.6 |
| Human | 94.5 | 94.8 |
| TransFuser / LTF | *not run* | 84.0 / 83.8 |
| Hydra-MDP (winner) | *not run* | 91.3 |

Warmup ≠ navtest, but the three blind/Human numbers replicate. Do not claim beating TransFuser.

## Next

Privileged ladder is done. A deployable next step is still visual (recover lane geometry), which needs mini sensors. Write-up: *blind kinematics ≈ MLP; map ≫ boxes; leftover Human is path + speed, not detections.*
