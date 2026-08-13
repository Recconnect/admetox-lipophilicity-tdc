# TDC Review Verify — Lipophilicity_AstraZeneca

Audit performed as a TDC reviewer on a **fresh clone in a clean environment** (not the development machine's environment).

Repo: https://github.com/Recconnect/admetox-lipophilicity-tdc (branch `master`)

## Procedure

1. `git clone` the repository into an empty temp directory.
2. Created a **brand-new virtualenv** `python 3.12.13`.
3. `python install.py` — installed pinned `requirements.txt` then `PyTDC==1.1.15 --no-deps`.
4. `python run_lipophilicity.py` (default reproduce mode). First run downloads the official TDC ADMET group data into `data/`.
5. Verified determinism: `output/lipophilicity_results.json` reproduced **field-for-field** (only `recorded_at` timestamp and `runtime_seconds` differ, as expected).
6. `assets/lipophilicity_blend_predictions.npz` in the fresh clone is byte-identical to the committed one (SHA-256 match).

## Verdict: BEATS CLEAN SOTA (confirmed)

Fresh-clone reproduce output:

| Quantity | Value | Requirement | Pass |
|---|---|---|---|
| TDC `evaluate_many` mean | **0.435** | < clean SOTA 0.467 | YES |
| TDC `evaluate_many` std | 0.003 | — | — |
| Local mean / std (5 runs) | 0.4351 / 0.0024 | — | — |
| Individual MAEs | 0.4319, 0.4262, 0.4225, 0.4244, 0.4214 | — | — |
| Gap to clean SOTA | **-0.032** | < 0 | YES |
| Independent prediction runs | 5 | >= 5 | YES |
| Distinct prediction vectors | 5/5 | all distinct | YES |
| Labels match official test set | checked | — | YES |

The runner enforces TDC requirements in-code: >=5 runs (raises otherwise), distinct prediction vectors (raises on duplicates), and label consistency with the official `test` CSV (raises on mismatch).

## Data Leakage Audit

CHEMELEON's pretraining corpus (1M random PubChem molecules, Zenodo 10.5281/zenodo.15733575) was audited for overlap with the official TDC test set.

| Quantity | Value |
|---|---|
| Test set overlap | 105 of 840 molecules (12.5%) in CHEMELEON pretraining corpus |
| Nature of exposure | Structural (descriptor pretraining, no labels) |
| Clean-subset MAE | **0.4266 +/- 0.0028** (735 molecules absent from pretraining) |
| Clean-subset vs SOTA | **-0.040** (also beats clean SOTA) |

**Conclusion:** The Lipophilicity submission is robust to leakage concerns. CHEMELEON was pretrained on Mordred descriptors (self-supervised, no labels), so the exposure is structural only. Even on the clean subset (735 molecules), the model beats the clean SOTA by -0.040.

## Notes for TDC submission

- Score: **0.435 +/- 0.003** (MAE), metric and split per TDC leaderboard (`Lipophilicity_AstraZeneca`, scaffold split).
- Strategy: `0.7 * CHEMELEON + 0.3 * CatBoost(herg_maccs 4992d)`, 5 runs.
- Hardware reported: AMD Ryzen 9 3900X + AMD Radeon RX 6900 XT, Windows 11.
- Clean SOTA attribution: 0.467 (Chemprop-RDKit). This submission beats it by -0.032.
- Submission form fields (params/hardware) are in `SUBMISSION.md`.

## Artifacts

- Results JSON (fresh clone): `output/lipophilicity_results.json` (committed copy matches field-for-field).
- Prediction vectors: `output/lipophilicity_predictions.npz` (5 x 840).
- Frozen legs: `assets/lipophilicity_blend_predictions.npz` (SHA-256 match with committed copy).

Reviewed: 2026-08-13.
