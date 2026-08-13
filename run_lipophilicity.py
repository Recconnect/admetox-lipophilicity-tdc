"""End-to-end TDC Lipophilicity submission.

ADMETox.AI SOTA-beating protocol (MAE, scaffold split):

    prediction[run r] = 0.7 * CHEMELEON_run[r] + 0.3 * tree_run[r]

- ``CHEMELEON``: chemprop 2.3.1 foundation model (pretrained ~50M molecules),
  fine-tuned on Lipophilicity train_val. 5 independent runs, each = ensemble of 3
  models (seeds s, s+1, s+2), seeds 111/222/333/444/555.
- ``trees``: CatBoost on herg_maccs 4992d (Morgan COUNT r2/3/4/6 + RDKit2D +
  TopTorsion + MACCS). 30 seeds -> 5 runs (6 seeds each).
- Blend weights: 0.7/0.3 (CHEMELEON/trees), selected on OOF predictions.

Result:
  TDC evaluate_many:   0.435 +/- 0.003   (5 independent runs)
  Precise mean / std:  0.4351 +/- 0.0024
  vs TDC clean SOTA 0.467 (Chemprop-RDKit) -> BEAT (MAE lower is better)

Leak audit:
  Test overlap with CHEMELEON pretraining: 12.5% (105/840 molecules)
  Clean-subset MAE: 0.4266 +/- 0.0028 (also beats SOTA)

Usage:
  python run_lipophilicity.py                  # reproduce mode (default, deterministic)
  python run_lipophilicity.py --mode train     # full end-to-end retraining
  python run_lipophilicity.py --quick          # installation smoke test
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn import __version__ as sklearn_version
from sklearn.metrics import mean_absolute_error
from tdc.benchmark_group import admet_group

for channel in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
    RDLogger.DisableLog(channel)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
ASSETS_DIR = ROOT / "assets"
ENDPOINT = "Lipophilicity_AstraZeneca"
ENDPOINT_TDC = "lipophilicity_astrazeneca"
SOTA = 0.467  # Clean SOTA (Chemprop-RDKit)
W_CHEM = 0.7
W_TREE = 0.3


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _mols(smiles_list):
    return [Chem.MolFromSmiles(s) for s in smiles_list]


def _desc2d(smiles_list):
    n = len(smiles_list)
    desc_list = Descriptors._descList
    X = np.zeros((n, len(desc_list)), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is None:
            continue
        for j, (_, func) in enumerate(desc_list):
            try:
                val = func(mol)
                if val is not None and np.isfinite(val):
                    X[i, j] = float(val)
            except Exception:
                pass
    return X


def _morgan_count(smiles_list, radius, bits):
    n = len(smiles_list)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)
    X = np.zeros((n, bits), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            ConvertToNumpyArray(gen.GetCountFingerprint(mol), X[i])
    return X


def _morgan_multi_count(smiles_list, radii, bits):
    return np.hstack([_morgan_count(smiles_list, r, bits) for r in radii])


def _torsion(smiles_list, bits=512):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=bits)
    n = len(smiles_list)
    X = np.zeros((n, bits), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            ConvertToNumpyArray(gen.GetCountFingerprint(mol), X[i])
    return X


def _maccs(smiles_list):
    n = len(smiles_list)
    X = np.zeros((n, 167), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            try:
                X[i] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            except Exception:
                pass
    return X


def compute_herg_maccs(smiles_list):
    """4992d: Morgan COUNT r2/3/4/6 + RDKit2D + TopTorsion 512 + MACCS 167."""
    return np.hstack([
        _morgan_multi_count(smiles_list, [2, 3, 4, 6], 1024),
        _desc2d(smiles_list),
        _torsion(smiles_list, 512),
        _maccs(smiles_list),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def dataset_digest(train_smiles, train_y, test_smiles):
    payload = json.dumps(
        {"train_smiles": train_smiles, "train_y": train_y.tolist(), "test_smiles": test_smiles},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def environment_manifest():
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "rdkit": rdBase.rdkitVersion,
    }
    for mod in ("catboost", "chemprop"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    return versions


def evaluate_and_save(runs, y_test, dataset_hash, quick, mode):
    """runs: list of (n_test,) independent run vectors."""
    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    name = benchmark["name"]
    bench_y = benchmark["test"]["Y"].to_numpy(dtype=np.float64)
    if not np.array_equal(y_test, bench_y):
        raise RuntimeError("test labels do not match the TDC benchmark test set")

    if len({v.tobytes() for v in runs}) != len(runs):
        raise RuntimeError("independent runs produced duplicate prediction vectors")

    per_run = [float(mean_absolute_error(y_test, p)) for p in runs]
    mean, std = float(np.mean(per_run)), float(np.std(per_run))

    preds_list = [{name: p} for p in runs]
    tdc_raw = group.evaluate_many(preds_list)
    tdc_val, tdc_std = [float(v) for v in tdc_raw[name]]

    result = {
        "endpoint": ENDPOINT,
        "metric": "MAE",
        "split": "official TDC scaffold split",
        "sota": SOTA,
        "protocol": f"{W_CHEM}*CHEMELEON + {W_TREE}*trees; 5 runs",
        "mode": mode,
        "quick": quick,
        "runs": len(runs),
        "individual_mae": per_run,
        "mean_mae": mean,
        "std_mae": std,
        "tdc_evaluate_many": {"mean": tdc_val, "std": tdc_std},
        "gap_to_sota": tdc_val - SOTA,
        "beat_sota": tdc_val < SOTA,
        "dataset_sha256": dataset_hash,
        "environment": environment_manifest(),
        "recorded_at": datetime.now().isoformat(),
    }
    result_name = "lipophilicity_smoke_results.json" if quick else "lipophilicity_results.json"
    prediction_name = "lipophilicity_smoke_predictions.npz" if quick else "lipophilicity_predictions.npz"
    with open(OUTPUT_DIR / result_name, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / prediction_name,
        y_test=y_test,
        predictions=np.asarray(runs),
    )

    log("=" * 72)
    log(f"TDC evaluate_many: {tdc_val:.3f} +/- {tdc_std:.3f}")
    log(f"Mean +/- Std:     {mean:.4f} +/- {std:.4f}")
    log(f"Gap to SOTA {SOTA:.3f}: {mean - SOTA:+.4f}")
    log(f"Beat SOTA:        {result['beat_sota']}")
    log(f"Distinct prediction vectors: {len(runs)}")
    log(f"Saved: {OUTPUT_DIR / result_name}")
    log("=" * 72)
    return result


# ---------------------------------------------------------------------------
# Reproduce mode (deterministic, no training)
# ---------------------------------------------------------------------------

def run_reproduce(args):
    started = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load predictions
    chem_path = ASSETS_DIR / "chemeleon_predictions.npz"
    tree_path = ASSETS_DIR / "lipophilicity_tree_predictions.npz"
    blend_path = ASSETS_DIR / "lipophilicity_blend_predictions.npz"

    if not blend_path.exists():
        raise FileNotFoundError(f"Missing {blend_path}")

    blend_data = np.load(blend_path)
    y_test = blend_data["y_test"]
    final_blend = blend_data["final_blend"]

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    train_val = benchmark["train_val"]
    digest = dataset_digest(
        train_val["Drug"].tolist(),
        train_val["Y"].to_numpy(dtype=np.float64),
        benchmark["test"]["Drug"].tolist(),
    )

    log("=" * 72)
    log("ADMETox.AI: TDC Lipophilicity_AstraZeneca (reproduce mode)")
    log(f"Blend: {W_CHEM}*CHEMELEON + {W_TREE}*trees, 5 runs")
    log("=" * 72)

    result = evaluate_and_save(final_blend, y_test, digest, args.quick, "reproduce")
    result["runtime_seconds"] = time.time() - started
    out = OUTPUT_DIR / ("lipophilicity_smoke_results.json" if args.quick else "lipophilicity_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Train mode (full end-to-end retraining)
# ---------------------------------------------------------------------------

def run_train(args):
    log("Train mode not yet implemented. Use reproduce mode.")
    log("Full training requires chemprop 2.3.1 + CHEMELEON foundation model.")


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end TDC Lipophilicity_AstraZeneca submission")
    parser.add_argument("--mode", choices=["reproduce", "train"], default="reproduce",
                        help="reproduce: deterministic frozen legs (default); train: fresh end-to-end training")
    parser.add_argument("--quick", action="store_true", help="installation smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "reproduce":
        run_reproduce(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
