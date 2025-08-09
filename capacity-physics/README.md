# capacity-physics

Early-stage, reproducible research pipeline for a computational-capacity model of time dilation. Micro fits only; macro validation without tuning.

---

**Scientific honesty box (paste verbatim):**

Status: Exploratory. Our current strong/weak-field checks are consistency tests. The constitutive B(N) is derived from micro DoF data (no GR tuning) and then frozen. All macro comparisons use this frozen form and report baselines (AIC/BIC). Code is fully reproducible; manifests contain hashes of inputs/configs. Any prediction is stated with a number, uncertainty, and where to test it.

---

## How to run

1. Place your DoF CSV in `data/thermo_runs/`.
2. Update/create a config in `configs/`.
3. Run scripts in order: `fit_micro.py` → `run_checks.py` → `run_ppn.py` → `run_geodesics.py`.

## Outputs

`reports/` contains plots, CSVs, baselines, and `manifest.json` with hashes.

## Prediction file

Add `PREDICTIONS.md` with one frozen, falsifiable number (domain, method, uncertainty).

---

## Directory structure

See instruction sheet for full details.
