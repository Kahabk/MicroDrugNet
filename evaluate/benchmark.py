"""
MicroDrugNet - Benchmarking wrapper

Supports both the older CLI:
  python evaluate/benchmark.py --data data/processed/dataset.pkl --ckpt checkpoints/best_model.pt

And the current real-data pipeline built around training_dataset.csv.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_root_evaluate_module():
    eval_path = ROOT / "evaluate.py"
    spec = importlib.util.spec_from_file_location("root_evaluate_module", eval_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL = _load_root_evaluate_module()


def _resolve_data_path(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested))
    candidates.extend([
        ROOT / "data/processed/training_dataset.csv",
        ROOT / "data/processed/synthetic_dataset.pkl",
    ])

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find a benchmark dataset. Searched:\n" + searched
    )



def _load_dataframe(data_path: Path, n_taxa: int, seed: int) -> pd.DataFrame:
    if data_path.suffix.lower() == ".csv":
        print(f"Loading real dataset from: {data_path}")
        return pd.read_csv(data_path)

    print(f"Loading legacy pickle dataset from: {data_path}")
    records = EVAL.load_dataset(str(data_path))

    rows = []
    for r in records:
        row = {
            "drug_name": r.get("drug_name", "unknown"),
            "smiles": r.get("smiles", ""),
            "bioavailability": float(r["bioavailability"]),
            "response_class": int(r.get("response_class", r.get("response", 0))),
            "toxicity": float(r.get("toxicity", 0.0)),
        }
        micro = r["microbiome"]
        if hasattr(micro, "cpu"):
            micro = micro.cpu().numpy()
        for i in range(n_taxa):
            row[f"taxa_{i}"] = float(micro[i]) if i < len(micro) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)



def main():
    parser = argparse.ArgumentParser(description="MicroDrugNet benchmark wrapper")
    parser.add_argument("--data", default="data/processed/dataset.pkl")
    parser.add_argument("--ckpt", default="checkpoints/best_model.pt")
    parser.add_argument("--n-taxa", dest="n_taxa", type=int, default=500)
    parser.add_argument("--test-frac", dest="test_frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/benchmark_results.json")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data)
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    df = _load_dataframe(data_path, n_taxa=args.n_taxa, seed=args.seed)

    group_col = EVAL._choose_group_column(df)
    if group_col is not None:
        train_df, test_df = EVAL._group_shuffle_split(df, group_col, args.test_frac, args.seed)
        print(
            f"Grouped benchmark split on '{group_col}': "
            f"train={len(train_df)} test={len(test_df)} | "
            f"groups={train_df[group_col].nunique()}/{test_df[group_col].nunique()}"
        )
    else:
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=args.test_frac,
            stratify=df["response_class"],
            random_state=args.seed,
        )
        print(f"Row-wise benchmark split: train={len(train_df)} test={len(test_df)}")

    EVAL.run_benchmarks(
        train_df=train_df,
        test_df=test_df,
        n_taxa=args.n_taxa,
        checkpoint_path=str(checkpoint_path),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
