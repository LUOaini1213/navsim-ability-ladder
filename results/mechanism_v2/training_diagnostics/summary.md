# Verified core training diagnostics

Full-budget last-checkpoint training diagnostics only; no PDMS result or deployment claim

All 18 runs: 13,000 optimizer updates; last checkpoint; parameters max/min = 1.0773.

Entries are three-seed mean [min, max], not confidence intervals.

| condition | parameters | ADE (m) | coordinate L1 (m) | heading L1 (rad) | cached forward (ms) | reversal max (m/rad) |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 583,192 | 0.6387 [0.6316, 0.6440] | 0.3698 [0.3664, 0.3727] | 0.0301 [0.0283, 0.0317] | 0.2854 [0.2207, 0.3395] | 0.0000 [0.0000, 0.0000] |
| M1 | 628,248 | 0.7078 [0.7039, 0.7132] | 0.4116 [0.4095, 0.4147] | 0.0318 [0.0310, 0.0335] | 0.2182 [0.1431, 0.3402] | 7.2512 [6.2593, 8.4490] |
| M2 | 628,248 | 0.7281 [0.7201, 0.7425] | 0.4226 [0.4178, 0.4313] | 0.0344 [0.0338, 0.0355] | 0.4087 [0.2021, 0.5542] | 7.2920 [6.2014, 8.6083] |
| M3 | 584,216 | 0.6120 [0.5997, 0.6205] | 0.3567 [0.3493, 0.3620] | 0.0283 [0.0279, 0.0285] | 0.3447 [0.1839, 0.4634] | 0.0000 [0.0000, 0.0000] |
| M4 | 617,368 | 0.6352 [0.6330, 0.6369] | 0.3690 [0.3683, 0.3697] | 0.0307 [0.0288, 0.0319] | 0.6133 [0.2948, 1.2292] | 0.0000 [0.0000, 0.0000] |
| M5 | 617,368 | 0.6423 [0.6358, 0.6492] | 0.3721 [0.3685, 0.3757] | 0.0305 [0.0287, 0.0334] | 0.4946 [0.3177, 0.7056] | 0.0000 [0.0000, 0.0000] |

## Interpretation limits

- Validation imitation losses are not official driving metrics; checkpoint selection was not based on these losses
- Min/max describe three fitted seeds, not confidence intervals or population training uncertainty
- CPU timing uses one cached tensor scene, 10 warm-ups and 200 forwards, two threads, on a shared machine; extraction, map routing, sensors, transfer and scoring are excluded
- Reversal compares up to the first 512 fixed validation inputs; concat sensitivity is a diagnostic, not a shuffled-training comparison
- Pose-difference maximum mixes coordinate metres and raw-heading radians; it is not a geometric distance

The JSON retains individual seed values, all five losses, timing and history/checkpoint hashes.
