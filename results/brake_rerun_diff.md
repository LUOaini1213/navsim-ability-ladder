# privileged_brake_mini: the two runs, scene by scene

Run 1 = `2026.08.20.12.52.28` (used in every report), run 2 = `2026.08.20.12.55.16`. Identical code and configuration snapshots (the hydra `code/` folders differ only in `output_dir`); same metric cache; 563 scenes in both.

| | run 1 | run 2 |
|---|---:|---:|
| mean PDMS | 0.6020 | 0.5926 |
| scenes with a different score | 130 of 563 | |
| run 2 higher / lower | 34 / 96 | |

## Which sub-score moves

| sub-score | scenes where it differs |
|---|---:|
| NC | 30 |
| DAC | 10 |
| EP | 127 |
| TTC | 36 |
| C | 31 |
| DDC | 5 |
| PDMS | 130 |

Ego progress differs in almost every affected scene, i.e. the *trajectory* itself differed between the runs, not only its scoring. With identical code and inputs that points upstream of the agent — at the scene loading / annotation order the agent consumes, or at the parallel scorer — rather than at a parameter change. It has not been pinned down; every report keeps using run 1 and says so.


## Largest per-scene differences

| token | run 1 | run 2 | NC 1→2 | DAC 1→2 | EP 1→2 |
|---|---:|---:|---|---|---|
| `2111b648fcba5bb7` | 1.000 | 0.000 | 1.00→0.00 | 1.00→1.00 | 1.00→0.00 |
| `2d63eaba6813539f` | 1.000 | 0.000 | 1.00→0.00 | 1.00→1.00 | 1.00→0.00 |
| `3ebe4c8a20155459` | 1.000 | 0.000 | 1.00→0.00 | 1.00→1.00 | 1.00→0.00 |
| `47fc4cd82c45583c` | 1.000 | 0.000 | 1.00→0.00 | 1.00→1.00 | 1.00→0.00 |
| `7e1f829a0de95258` | 1.000 | 0.000 | 1.00→0.00 | 1.00→1.00 | 1.00→0.00 |
| `bd67b68fea295e96` | 0.000 | 1.000 | 1.00→1.00 | 0.00→1.00 | 0.00→1.00 |
| `aa9a9fdb89275acb` | 0.000 | 0.952 | 1.00→1.00 | 0.00→1.00 | 0.00→0.88 |
| `9e0633fbdeac55bb` | 0.931 | 0.000 | 1.00→0.00 | 1.00→1.00 | 0.83→0.00 |
| `b2f19dc9ecc052b4` | 0.000 | 0.851 | 0.00→1.00 | 1.00→1.00 | 0.00→0.64 |
| `f9b38490d7155d84` | 0.000 | 0.833 | 0.00→1.00 | 1.00→1.00 | 0.00→1.00 |
| `d9fac9fdd2bd5036` | 0.000 | 0.830 | 0.00→1.00 | 1.00→1.00 | 0.00→0.59 |
| `c18771a3868f5868` | 0.978 | 0.168 | 1.00→1.00 | 1.00→1.00 | 0.95→0.40 |
| `d84f5656f4f753e4` | 0.000 | 0.801 | 0.00→1.00 | 1.00→1.00 | 0.00→0.52 |
| `a64cd79798845d53` | 0.000 | 0.776 | 0.50→1.00 | 0.00→1.00 | 0.00→0.46 |
| `8ce2cf49a1955788` | 0.000 | 0.763 | 0.50→1.00 | 0.00→1.00 | 0.00→0.43 |
