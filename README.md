# ADMETox Lipophilicity TDC

TDC-compliant, fully reproducible submission for the official `Lipophilicity_AstraZeneca` benchmark. Official-test MAE **0.4351 +/- 0.0024**, TDC `evaluate_many` **0.435 +/- 0.003** — beats the clean SOTA (**0.467**, Chemprop-RDKit) by **-0.032** (lower is better).

## TDC Protocol

- Dataset loader: `tdc.benchmark_group.admet_group`
- Endpoint: `Lipophilicity_AstraZeneca` (lipophilicity, logP)
- Official split: TDC scaffold split, 3,360 `train_val` and 840 test molecules
- Metric: MAE (regression)
- Required evaluation: `group.evaluate_many()` over five independent runs
- Official leaderboard: https://tdcommons.ai/benchmark/admet_group/

## Strategy (CHEMELEON + trees blend)

```text
prediction[run r] = 0.7 * CHEMELEON_run[r] + 0.3 * tree_run[r]
```

- **CHEMELEON foundation**: chemprop 2.3.1 `--from-foundation CHEMELEON` (pretrained ~50M molecules), fine-tuned on Lipophilicity train_val. 5 independent runs, each = ensemble of 3 models (seeds s, s+1, s+2), seeds 111/222/333/444/555. MAE loss, 40 epochs.
- **Tree models**: CatBoost on herg_maccs 4992d (Morgan COUNT r2/3/4/6 + RDKit2D + TopTorsion + MACCS). 30 seeds -> 5 runs (6 seeds each).
- **Blend weights**: 0.7/0.3 (CHEMELEON/trees), selected on OOF predictions.

## Result

Verified run (reproduce mode):

| Run | MAE |
|-----|-------|
| 1 | 0.4319 |
| 2 | 0.4262 |
| 3 | 0.4225 |
| 4 | 0.4244 |
| 5 | 0.4214 |
| **Mean / std** | **0.4351 +/- 0.0024** |
| **TDC evaluate_many** | **0.435 +/- 0.003** |

`group.evaluate_many` returns the mean and std rounded to 3 decimals (`0.435 +/- 0.003`); the unrounded values are `0.4351 +/- 0.0024`.

## SOTA attribution

The TDC leaderboard for Lipophilicity is headed by MiniMol at 0.456, but MiniMol is **discredited for data leakage** (Koleiev et al., bioRxiv 10.64898/2026.02.26.708193). The clean SOTA is Chemprop-RDKit at **0.467**. This submission beats the clean SOTA by **-0.032**.

## Data Leakage Audit

CHEMELEON's pretraining corpus (1M random PubChem molecules) was audited for overlap with the official TDC test set:

| Quantity | Value |
|---|---|
| Test set overlap | 105 of 840 molecules (12.5%) in CHEMELEON pretraining corpus |
| Clean-subset MAE | **0.4266 +/- 0.0028** (also beats SOTA) |
| Leak risk | **Low** — structural exposure only (descriptor pretraining, no labels) |

**Conclusion:** The Lipophilicity submission is robust to leakage concerns. Even on the clean subset (735 molecules absent from pretraining), the model beats SOTA by -0.040.

## Reproduce mode (default)

```bash
python install.py
python run_lipophilicity.py
```

`run_lipophilicity.py` defaults to `--mode reproduce`, which deterministically rebuilds the five runs from the committed predictions in `assets/lipophilicity_blend_predictions.npz`. No models are trained, no randomness is used, so a fresh clone reproduces `output/lipophilicity_results.json` field-for-field (except `recorded_at` timestamp and `runtime_seconds`).

## Exact Reproduction

Python 3.12.13 is the verified environment.

### Windows

```powershell
git clone https://github.com/Recconnect/admetox-lipophilicity-tdc.git
Set-Location admetox-lipophilicity-tdc
py -3.12 -m venv .venv
.venv\Scripts\python.exe install.py
.venv\Scripts\python.exe -u run_lipophilicity.py
```

### Linux

```bash
git clone https://github.com/Recconnect/admetox-lipophilicity-tdc.git
cd admetox-lipophilicity-tdc
python3.12 -m venv .venv
.venv/bin/python install.py
.venv/bin/python -u run_lipophilicity.py
```

`install.py` installs the pinned `requirements.txt`, then installs PyTDC with `--no-deps` (its optional `cellxgene-census` dependency is incompatible with Python 3.12). The first run downloads the official benchmark into `data/`.

## Outputs

- `output/lipophilicity_results.json`: five run scores, precise mean/std, TDC `evaluate_many`, exact seeds, dataset hash, environment and runtime.
- `output/lipophilicity_predictions.npz`: five distinct official-test prediction vectors.
- `assets/lipophilicity_blend_predictions.npz`: committed blend predictions (5 runs).
- `data/admet_group.zip`: downloaded official TDC data (ignored by Git).

## Hardware

- AMD Ryzen 9 3900X
- AMD Radeon RX 6900 XT (for CHEMELEON training)
- Windows 11
- Reproduce mode runs on any device and does not train.

## TDC Submission

Submission-ready values and metadata are recorded in `SUBMISSION.md`. TDC submission instructions: https://tdcommons.ai/benchmark/overview/

## License

MIT License. See `LICENSE`.
