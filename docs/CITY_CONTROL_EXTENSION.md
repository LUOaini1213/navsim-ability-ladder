# Same-city donor extension

This design responds to the input-only cross-city audit. It is an auxiliary,
conditional follow-up, not a rewrite of the frozen core or a completed experiment.
The generator does not read outcome arrays. No auxiliary PDMS result exists yet.

## Fixed comparison

| condition | model/input | training status |
|---|---|---|
| C1 | base M1, real nearest-eight concat | reuse only with unchanged complete train cohort/normalization |
| C2 | M2 configuration, same-city cross-session donors | six auxiliary models in total with C5; three seeds each |
| C4 | base M4, real masked set | same reuse restriction as C1 |
| C5 | M5 configuration, same-city cross-session donors | new training, same fixed 13,000 updates |

Donors remain in their partition and city and must come from another original
driving session. Match command, speed bin and count bin; record any relaxation
of count, speed or command. City/session independence is never relaxed.
The sampling chooses a donor session before a donor scene, as in the base design.

The input audit shows 12/26/70/22 training sessions in Singapore/Boston/Las Vegas/
Pittsburgh. Each supports cross-session city donors without changing the full
training cohort. Validation and confirmation each contain only one Singapore
session, so Singapore cannot enter their strict same-city comparison. Its
omission is input-driven and must be disclosed with the exact resulting counts.
Navtest has at least two sessions per city, but remains exploratory.

`prepare_city_control.py` freezes a separate protocol and donor file, including
eligible token order and omitted-city coverage. It does not alter base files or
launch training. Repeated preparation must reproduce the same mappings.
After training, auxiliary normalizers must match those of the full-data anchors.
If they do not, stop and investigate; never just compare mismatched cohorts.

The actual frozen input audit now passes (`results/audit_v2/city_inputs.json`).
Training is unchanged at 83,099 scenes/130 sessions. Common validation coverage is
10,312 scenes/15 sessions (762 Singapore scenes excluded); confirmation is
8,379/15 (736 excluded). Exploratory navtest retains all 12,146 scenes/44 sessions.
Every actual donor is same-city/different-session. The base helper's inherited
text saying city is unconstrained is inapplicable here and explicitly corrected
in the separate audit note, without changing the pre-outcome frozen generator.
These input facts are not six trained auxiliary models or a PDMS comparison.

## Analysis and confirmation

Three fixed contrasts: C1-C2, C4-C5 and (C4-C5)-(C1-C2). Report per-seed effects,
session/log intervals, component metrics, eligible scenes/sessions and matching
coverage. Auxiliary multiplicity intervals do not confer an unqualified global
confirmatory guarantee on the adaptive study. Outcome-triggered follow-up and
the original hypothesis-generating navtest use must remain explicit.

Every participating model is evaluated on exactly the registered eligible tokens,
not on its most favorable subset. Reused base predictions must be hash-bound to
their source receipt and filtered by the frozen cohort, not re-labelled without
provenance. Six auxiliary models are not an 18-model core: a separate exporter
must gate confirmation on both all 18 base models and all six auxiliary models.
No confirmation-gate bypass is authorized by this design.

Same-city swapping still destroys route/actor compatibility. It is a stronger
distribution control, not a realistic intervention on traffic or a proof of
causal safety. If an alignment interpretation remains unsupported, withhold it
and deliver the planned bounded technical report.
