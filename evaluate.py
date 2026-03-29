"""
MicroDrugNet — Evaluation & Benchmarking
=========================================
Compares four models on the HELD-OUT test set:

  1. RandomForest       – Morgan fingerprint + log1p(microbiome)
  2. MLP_Baseline       – same features, scikit-learn MLP
  3. DrugOnly_GNN       – MicroDrugNet with microbiome ZEROED OUT
                          (proves microbiome fusion actually helps)
  4. MicroDrugNet       – full model, loads best_model.pt checkpoint

Metrics reported per model:
  - response_auroc      (primary metric, macro one-vs-rest)
  - response_accuracy
  - response_f1_macro
  - bioavail_mae
  - bioavail_rmse
  - bioavail_r2
  - toxicity_auroc      (binary, threshold=0.5)

IMPORTANT:
  - Baselines are trained on the TRAINING split, evaluated on the TEST split.
  - MicroDrugNet loads its checkpoint (trained separately by train.py).
  - The test split is NEVER seen during training.
  - All metrics are computed on identical test samples for fair comparison.

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pt \\
                     --n_taxa 500 --n_samples 5000
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from microdrug.data_utils import (
    generate_synthetic_dataset,
    MicroDrugDataset,
    collate_fn,
)
from microdrug.model import MicroDrugNet
from torch.utils.data import DataLoader


def _choose_group_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["microbiome_sample_id", "sample_id", "subject_id"]:
        if col in df.columns:
            return col
    return None



def _group_shuffle_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 Morgan fingerprint. Returns zero vector on failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return np.zeros(n_bits, dtype=np.float32)

        # Prefer the newer RDKit MorganGenerator API to avoid deprecation noise.
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            fp = GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(mol)
        except Exception:
            from rdkit.Chem import AllChem
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)

        return np.array(fp, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(n_bits, dtype=np.float32)


def _build_features(
    df: pd.DataFrame,
    taxa_cols: List[str],
    fp_bits: int = 2048,
) -> np.ndarray:
    """
    Build feature matrix for sklearn baselines:
      [Morgan fingerprint (2048) | log1p(microbiome) (n_taxa)]
    """
    fps  = np.array([_morgan_fp(s, fp_bits) for s in df["smiles"]])
    micro = np.log1p(df[taxa_cols].to_numpy(dtype=np.float32))
    return np.concatenate([fps, micro], axis=1)


def _compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    bio_true: np.ndarray,
    bio_pred: np.ndarray,
    tox_true: np.ndarray,
    tox_pred: np.ndarray,
    model_name: str,
) -> Dict:
    """Compute all metrics from numpy arrays. No torch tensors here."""
    y_pred = y_proba.argmax(axis=1)

    # Response AUROC (macro OvR)
    try:
        auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                    average="macro"))
    except ValueError as e:
        print(f"  WARNING: AUROC skipped ({e})")
        auroc = float("nan")

    acc    = float(accuracy_score(y_true, y_pred))
    f1     = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mae    = float(mean_absolute_error(bio_true, bio_pred))
    rmse   = float(np.sqrt(((bio_true - bio_pred) ** 2).mean()))

    ss_res = ((bio_true - bio_pred) ** 2).sum()
    ss_tot = ((bio_true - bio_true.mean()) ** 2).sum()
    r2     = float(1.0 - ss_res / (ss_tot + 1e-8))

    # Toxicity AUROC (binary at 0.5 threshold)
    tox_binary = (tox_true > 0.5).astype(int)
    try:
        tox_auroc = float(roc_auc_score(tox_binary, tox_pred))
    except ValueError:
        tox_auroc = float("nan")   # only one class in this split

    return {
        "model":            model_name,
        "response_auroc":   auroc,
        "response_accuracy": acc,
        "response_f1_macro": f1,
        "bioavail_mae":     mae,
        "bioavail_rmse":    rmse,
        "bioavail_r2":      r2,
        "toxicity_auroc":   tox_auroc,
    }


def _print_metrics(m: Dict):
    auroc = m["response_auroc"]
    goal  = "  <-- TARGET MET" if (not np.isnan(auroc) and auroc > 0.82) else ""
    print(f"\n  Model: {m['model']}")
    print(f"    response_auroc   : {auroc:.4f}{goal}")
    print(f"    response_accuracy: {m['response_accuracy']:.4f}")
    print(f"    response_f1_macro: {m['response_f1_macro']:.4f}")
    print(f"    bioavail_mae     : {m['bioavail_mae']:.4f}")
    print(f"    bioavail_rmse    : {m['bioavail_rmse']:.4f}")
    print(f"    bioavail_r2      : {m['bioavail_r2']:.4f}")
    print(f"    toxicity_auroc   : {m['toxicity_auroc']:.4f}")


# ─────────────────────────────────────────────────────────────────
# BASELINE 1: RANDOM FOREST
# ─────────────────────────────────────────────────────────────────

class RandomForestBaseline:
    """
    Trains two separate Random Forests:
      - response classifier  (multi-class, 3 classes)
      - bioavailability regressor

    Feature vector: Morgan ECFP4 (2048 bits) + log1p(OTU profile)
    """

    def __init__(self, n_estimators: int = 200, n_taxa: int = 500):
        self.n_taxa    = n_taxa
        self.taxa_cols = [f"taxa_{i}" for i in range(n_taxa)]
        self.scaler    = StandardScaler()

        self.resp_model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        self.bio_model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=42,
        )
        self.tox_model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, train_df: pd.DataFrame):
        print("  Training RandomForest ...")
        X = self.scaler.fit_transform(_build_features(train_df, self.taxa_cols))
        self.resp_model.fit(X, train_df["response_class"])
        self.bio_model.fit(X,  train_df["bioavailability"])
        self.tox_model.fit(X,  train_df["toxicity"])
        return self

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = self.scaler.transform(_build_features(test_df, self.taxa_cols))
        return {
            "response_proba":  self.resp_model.predict_proba(X),
            "bioavailability": self.bio_model.predict(X),
            "toxicity":        self.tox_model.predict(X),
        }


# ─────────────────────────────────────────────────────────────────
# BASELINE 2: MLP
# ─────────────────────────────────────────────────────────────────

class MLPBaseline:
    """
    scikit-learn MLP. Same feature vector as Random Forest.
    """

    def __init__(self, n_taxa: int = 500):
        self.n_taxa    = n_taxa
        self.taxa_cols = [f"taxa_{i}" for i in range(n_taxa)]

        self.resp_model = Pipeline([
            ("sc",  StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])
        self.bio_model = Pipeline([
            ("sc",  StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])
        self.tox_model = Pipeline([
            ("sc",  StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ])

    def fit(self, train_df: pd.DataFrame):
        print("  Training MLP ...")
        X = _build_features(train_df, self.taxa_cols)
        self.resp_model.fit(X, train_df["response_class"])
        self.bio_model.fit(X,  train_df["bioavailability"])
        self.tox_model.fit(X,  train_df["toxicity"])
        return self

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = _build_features(test_df, self.taxa_cols)
        return {
            "response_proba":  self.resp_model.predict_proba(X),
            "bioavailability": self.bio_model.predict(X),
            "toxicity":        self.tox_model.predict(X),
        }


# ─────────────────────────────────────────────────────────────────
# BASELINE 3: DRUG-ONLY ABLATION
# ─────────────────────────────────────────────────────────────────
# This is critical for proving that your microbiome fusion helps.
# It runs MicroDrugNet with the microbiome input ZEROED OUT.
# If DrugOnly_GNN scores 0.76 and MicroDrugNet scores 0.86,
# that delta is your novel contribution.

# (Implemented inside evaluate_microdrug by passing zero_microbiome=True)


# ─────────────────────────────────────────────────────────────────
# MICRODRUG EVALUATOR (used for both full model and drug-only ablation)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_microdrug(
    model: MicroDrugNet,
    test_df: pd.DataFrame,
    n_taxa: int,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    zero_microbiome: bool = False,
    model_name: str = "MicroDrugNet",
) -> Dict:
    """
    Evaluate a MicroDrugNet checkpoint on the test DataFrame.

    Parameters
    ----------
    model           : MicroDrugNet instance (architecture only; weights loaded here)
    test_df         : held-out test split
    n_taxa          : number of taxa features
    device          : torch device
    checkpoint_path : path to best_model.pt; if None uses random weights
    zero_microbiome : if True, zeroes the microbiome tensor (drug-only ablation)
    model_name      : label used in output table
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: epoch {ckpt['epoch']}  "
              f"(val AUROC was {ckpt['metrics'].get('val_response_auroc', 'n/a')})")
    else:
        print(f"  WARNING: checkpoint not found at '{checkpoint_path}'. "
              f"Using RANDOM WEIGHTS -- metrics will be meaningless (~0.33 AUROC).")

    model.eval().to(device)

    dataset = MicroDrugDataset(test_df, n_taxa=n_taxa, cache_graphs=True)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=False,
    )

    all_bio_pred, all_bio_true   = [], []
    all_resp_proba, all_resp_true = [], []
    all_tox_pred, all_tox_true   = [], []

    for batch in loader:
        drug_graph = batch["drug_graph"].to(device)
        microbiome = batch["microbiome"].to(device)
        condition_feat = batch["condition_feat"].to(device)

        if zero_microbiome:
            microbiome = torch.zeros_like(microbiome)

        preds = model(drug_graph, microbiome, condition_feat=condition_feat)

        all_bio_pred.append(preds["bioavailability"].cpu().numpy())
        all_bio_true.append(batch["bioavailability"].numpy())

        all_resp_proba.append(
            torch.softmax(preds["response_class"], dim=-1).cpu().numpy()
        )
        all_resp_true.append(batch["response_class"].numpy())

        all_tox_pred.append(preds["toxicity"].cpu().numpy())
        all_tox_true.append(batch["toxicity"].numpy())

    bio_pred   = np.concatenate(all_bio_pred).squeeze()
    bio_true   = np.concatenate(all_bio_true)
    resp_proba = np.concatenate(all_resp_proba)
    resp_true  = np.concatenate(all_resp_true)
    tox_pred   = np.concatenate(all_tox_pred).squeeze()
    tox_true   = np.concatenate(all_tox_true)

    return _compute_metrics(
        resp_true, resp_proba,
        bio_true,  bio_pred,
        tox_true,  tox_pred,
        model_name,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────

def run_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_taxa: int,
    checkpoint_path: str,
    output_path: str = "checkpoints/benchmark_results.json",
) -> pd.DataFrame:
    """
    Train all baselines on train_df.
    Evaluate all models on test_df.
    Print and save the results table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Train set: {len(train_df)}  |  Test set: {len(test_df)}")

    taxa_cols = [f"taxa_{i}" for i in range(n_taxa)]
    all_results: List[Dict] = []

    # ── 1. Random Forest ──────────────────────────────────────────
    rf = RandomForestBaseline(n_estimators=200, n_taxa=n_taxa)
    rf.fit(train_df)
    rf_preds = rf.predict(test_df)
    rf_metrics = _compute_metrics(
        test_df["response_class"].values,
        rf_preds["response_proba"],
        test_df["bioavailability"].values,
        rf_preds["bioavailability"],
        test_df["toxicity"].values,
        rf_preds["toxicity"],
        "RandomForest",
    )
    _print_metrics(rf_metrics)
    all_results.append(rf_metrics)

    # ── 2. MLP ────────────────────────────────────────────────────
    mlp = MLPBaseline(n_taxa=n_taxa)
    mlp.fit(train_df)
    mlp_preds = mlp.predict(test_df)
    mlp_metrics = _compute_metrics(
        test_df["response_class"].values,
        mlp_preds["response_proba"],
        test_df["bioavailability"].values,
        mlp_preds["bioavailability"],
        test_df["toxicity"].values,
        mlp_preds["toxicity"],
        "MLP_Baseline",
    )
    _print_metrics(mlp_metrics)
    all_results.append(mlp_metrics)

    # ── 3. Drug-only ablation (microbiome ZEROED) ─────────────────
    print("\n  Running drug-only ablation (microbiome zeroed) ...")
    model_drug_only = MicroDrugNet(n_taxa=n_taxa).to(device)
    drug_only_metrics = evaluate_microdrug(
        model    = model_drug_only,
        test_df  = test_df,
        n_taxa   = n_taxa,
        device   = device,
        checkpoint_path  = checkpoint_path,
        zero_microbiome  = True,
        model_name       = "DrugOnly_GNN",
    )
    _print_metrics(drug_only_metrics)
    all_results.append(drug_only_metrics)

    # ── 4. Full MicroDrugNet ──────────────────────────────────────
    print("\n  Running full MicroDrugNet ...")
    model_full = MicroDrugNet(n_taxa=n_taxa).to(device)
    full_metrics = evaluate_microdrug(
        model    = model_full,
        test_df  = test_df,
        n_taxa   = n_taxa,
        device   = device,
        checkpoint_path = checkpoint_path,
        zero_microbiome = False,
        model_name      = "MicroDrugNet",
    )
    _print_metrics(full_metrics)
    all_results.append(full_metrics)

    # ── Print final table ─────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    cols_ordered = [
        "model", "response_auroc", "response_accuracy", "response_f1_macro",
        "bioavail_mae", "bioavail_rmse", "bioavail_r2", "toxicity_auroc",
    ]
    results_df = results_df[cols_ordered]

    print("\n" + "=" * 80)
    print("  FINAL BENCHMARK TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    # Ablation delta
    mdn_row  = results_df[results_df["model"] == "MicroDrugNet"]
    drug_row = results_df[results_df["model"] == "DrugOnly_GNN"]
    if not mdn_row.empty and not drug_row.empty:
        delta = (mdn_row["response_auroc"].values[0]
                 - drug_row["response_auroc"].values[0])
        print(f"\n  Microbiome contribution (delta AUROC): +{delta:.4f}")
        print(f"  (This is your novel contribution to prove in the paper.)")

    mdn_auroc = (mdn_row["response_auroc"].values[0]
                 if not mdn_row.empty else 0.0)
    if mdn_auroc > 0.82:
        print(f"\n  TARGET MET: AUROC {mdn_auroc:.4f} > 0.82 -- publishable quality!")
    else:
        print(f"\n  Current AUROC: {mdn_auroc:.4f} | Target: > 0.82")
        print("  Keep training or use real data (ChEMBL + HMP).")

    # ── Save results ──────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  Results saved to: {output_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MicroDrugNet benchmark evaluation")
    parser.add_argument("--checkpoint",  default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--n_taxa",      type=int, default=500,
                        help="Number of microbiome taxa features")
    parser.add_argument("--n_samples",   type=int, default=5000,
                        help="Synthetic dataset size (ignored if --data_path given)")
    parser.add_argument("--data_path",   default="data/processed/training_dataset.csv",
                        help="Path to real CSV dataset (optional)")
    parser.add_argument("--output",      default="checkpoints/benchmark_results.json")
    parser.add_argument("--test_frac",   type=float, default=0.20)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # ── Load or generate dataset ──────────────────────────────────
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading real dataset from: {args.data_path}")
        df = pd.read_csv(args.data_path)
    else:
        print("No --data_path given. Generating synthetic dataset ...")
        df = generate_synthetic_dataset(
            n_samples=args.n_samples,
            n_taxa=args.n_taxa,
            seed=args.seed,
        )

    # ── Train / test split ───────────────────────────────────────
    # Use group-disjoint splitting when microbiome group ids are available.
    group_col = _choose_group_column(df)
    if group_col is not None:
        train_df, test_df = _group_shuffle_split(df, group_col, args.test_frac, args.seed)
        print(
            f"Grouped benchmark split on '{group_col}': "
            f"train={len(train_df)} test={len(test_df)} | "
            f"groups={train_df[group_col].nunique()}/{test_df[group_col].nunique()}"
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_frac,
            stratify=df["response_class"],
            random_state=args.seed,
        )
        print(f"Row-wise benchmark split: train={len(train_df)} test={len(test_df)}")

    run_benchmarks(
        train_df       = train_df,
        test_df        = test_df,
        n_taxa         = args.n_taxa,
        checkpoint_path= args.checkpoint,
        output_path    = args.output,
    )


if __name__ == "__main__":
    main()
