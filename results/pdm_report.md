# NAVSIM warmup_test_e2e PDM comparison

Split: `warmup_test_e2e` (same 563-scene cache as CV 0.233).

| agent | n | PDMS | NC | DAC | EP | TTC | C | score=0 | NC fail | DAC fail | TTC fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CV | 563 | 0.233 | 0.653 | 0.643 | 0.232 | 0.437 | 1.000 | 0.623 | 0.387 | 0.357 | 0.563 |
| Kinematic | 563 | 0.580 | 0.899 | 0.737 | 0.539 | 0.776 | 1.000 | 0.325 | 0.124 | 0.263 | 0.224 |
| MLP | 563 | 0.640 | 0.920 | 0.806 | 0.598 | 0.787 | 1.000 | 0.250 | 0.091 | 0.194 | 0.213 |
| PrivBrake | 563 | 0.602 | 0.914 | 0.766 | 0.532 | 0.821 | 0.982 | 0.291 | 0.105 | 0.234 | 0.179 |
| PrivMap | 563 | 0.786 | 0.960 | 0.934 | 0.703 | 0.888 | 0.964 | 0.101 | 0.043 | 0.066 | 0.112 |
| PrivMapKin | 563 | 0.802 | 0.955 | 0.950 | 0.745 | 0.865 | 0.988 | 0.094 | 0.046 | 0.050 | 0.135 |
| PrivMapGTSpd | 563 | 0.866 | 0.976 | 0.950 | 0.815 | 0.931 | 0.984 | 0.067 | 0.025 | 0.050 | 0.069 |
| PrivGTPathKin | 563 | 0.833 | 0.996 | 0.973 | 0.718 | 0.986 | 0.840 | 0.030 | 0.005 | 0.027 | 0.014 |
| Human | 563 | 0.945 | 1.000 | 0.998 | 0.875 | 0.996 | 0.996 | 0.002 | 0.000 | 0.002 | 0.004 |

## vs ConstantVelocity

- **CV**: ΔPDMS=+0.000, better than CV 0.0%
- **Kinematic**: ΔPDMS=+0.347, better than CV 60.7%
- **MLP**: ΔPDMS=+0.407, better than CV 68.4%
- **PrivBrake**: ΔPDMS=+0.369, better than CV 63.8%
- **PrivMap**: ΔPDMS=+0.553, better than CV 83.3%
- **PrivMapKin**: ΔPDMS=+0.568, better than CV 84.7%
- **PrivMapGTSpd**: ΔPDMS=+0.632, better than CV 92.2%
- **PrivGTPathKin**: ΔPDMS=+0.600, better than CV 92.0%
- **Human**: ΔPDMS=+0.711, better than CV 99.3%

## MLP failure families

- ok: 359
- DAC: 90
- TTC: 63
- NC: 51

Dumped trajectories: NC=802ffb33c2655bab, DAC=dace7f508e4b5070, TTC=8fb4110a350b5f17 (`results/figures/traj_warmup_fail_*.png`); the `ok=4b10d7d9e7465633` example was dumped in the workspace but is not committed

## How to read this

- **Human** is privileged GT future — upper bound, not a deployable planner.
- **CV** is naive (constant speed, heading 0). Low EP + TTC/NC because it cannot slow or turn.
- **Kinematic** uses accel + left/right command yaw. Tests how far kinematics go without sensors.
- **MLP** is still blind (no camera/LiDAR) but *learns* the mapping from (v,a,command) → 4s trajectory.
- **PrivBrake** is not deployable: kinematics + IDM brake from GT boxes.
- **PrivMap** / **PrivMapKin** / **PrivMapGTSpd**: centerline with IDM, kinematic, or Human arc-length.
- **PrivGTPathKin**: Human logged path with kinematic speed (geometry vs timing).
- Remaining MLP zeros are mostly NC/DAC/TTC: scenes that need perception, not better kinematics.