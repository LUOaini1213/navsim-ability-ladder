# Per-scene PDM scores, as written by NAVSIM's run_pdm_score.py

One CSV per agent (563 scenes each, columns token / valid / the five sub-scores / score), copied verbatim from the workspace run listed below. `token_log.csv` maps every scene token to its source log (from the metric-cache layout); `available_mini_logs.yaml` is the upstream train/val log split the clean ladder uses. Every table and confidence interval in this repository is recomputed from these files by `analysis/analyze_ci.py` and `analysis/analyze_clean_ladder.py`.

| file | workspace source | note |
|---|---|---|
| `cv_agent_mini.csv` | `exp/cv_agent_mini/2026.08.14.09.40.13/2026.08.14.09.41.27.csv` | first run (canonical) |
| `kinematic_agent_mini.csv` | `exp/kinematic_agent_mini/2026.08.20.12.38.45/2026.08.20.12.40.20.csv` | first run (canonical) |
| `privileged_brake_mini.csv` | `exp/privileged_brake_mini/2026.08.20.12.52.28/2026.08.20.12.54.17.csv` | first run (canonical) |
| `privileged_brake_mini_run2.csv` | `exp/privileged_brake_mini/2026.08.20.12.55.16/2026.08.20.12.57.23.csv` | second run, kept for the documented discrepancy; not used by any report |
| `mlp_agent_mini.csv` | `exp/mlp_agent_mini/2026.08.14.10.02.42/2026.08.14.10.04.18.csv` | first run (canonical) |
| `map_mlp_mini.csv` | `exp/map_mlp_mini/2026.09.02.02.28.36/2026.09.02.02.30.32.csv` | first run (canonical) |
| `map_mlp_reg_mini.csv` | `exp/map_mlp_reg_mini/2026.09.02.09.44.56/2026.09.02.09.46.38.csv` | first run (canonical) |
| `privileged_centerline_mini.csv` | `exp/privileged_centerline_mini/2026.08.20.13.08.10/2026.08.20.13.10.02.csv` | first run (canonical) |
| `privileged_centerline_kin.csv` | `exp/privileged_centerline_kin/2026.08.20.13.28.11/2026.08.20.13.30.03.csv` | first run (canonical) |
| `speed_mlp_mini.csv` | `exp/speed_mlp_mini/2026.09.02.09.46.47/2026.09.02.09.48.21.csv` | first run (canonical) |
| `speed_mlp_e200_mini.csv` | `exp/speed_mlp_e200_mini/2026.09.02.09.51.00/2026.09.02.09.52.33.csv` | first run (canonical) |
| `privileged_gtpath_kin.csv` | `exp/privileged_gtpath_kin/2026.08.20.13.31.46/2026.08.20.13.33.09.csv` | first run (canonical) |
| `privileged_centerline_gtspeed.csv` | `exp/privileged_centerline_gtspeed/2026.08.20.13.30.11/2026.08.20.13.31.39.csv` | first run (canonical) |
| `human_agent_mini.csv` | `exp/human_agent_mini/2026.08.20.12.40.48/2026.08.20.12.42.04.csv` | first run (canonical) |
