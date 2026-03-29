"""
MicroDrugNet — Preprocessing Pipeline
Converts raw data into training-ready tensors.
"""
import os, json, pickle, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data

RAW   = Path("data/raw")
PROC  = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)


# ── Drug graph conversion ──────────────────────────────────────────
ATOM_FEATURES = 9

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        int(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    nodes = [atom_features(a) for a in mol.GetAtoms()]
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        return None
    x = torch.tensor(nodes, dtype=torch.float)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei)


# ── Microbiome OTU normalisation ──────────────────────────────────
def normalise_otu(df: pd.DataFrame, n_taxa: int = 1000) -> np.ndarray:
    """CLR-normalise + select top-n_taxa by variance."""
    df = df.fillna(0).astype(float)
    # Select highest-variance taxa
    var = df.var(axis=0).nlargest(n_taxa).index
    df  = df[var]
    # Pad/trim to exactly n_taxa
    if df.shape[1] < n_taxa:
        pad = pd.DataFrame(
            np.zeros((len(df), n_taxa - df.shape[1])), index=df.index
        )
        df = pd.concat([df, pad], axis=1)
    arr = df.values[:, :n_taxa]
    # CLR transform
    arr = arr + 1e-6
    log = np.log(arr)
    clr = log - log.mean(axis=1, keepdims=True)
    return clr.astype(np.float32)


# ── Build interaction dataset ──────────────────────────────────────
def build_dataset(interaction_csv: str, smiles_col="smiles",
                  microbiome_col="sample_id", label_col="bioavailability",
                  otu_table: str = None, n_taxa: int = 1000):
    """
    interaction_csv: rows are drug-microbiome pairs with labels.
    otu_table: sample_id → OTU abundances.
    """
    df = pd.read_csv(interaction_csv)
    otu = pd.read_csv(otu_table, index_col=0) if otu_table else None

    records = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is None:
            continue
        if otu is not None and row[microbiome_col] in otu.index:
            micro = torch.tensor(
                normalise_otu(otu.loc[[row[microbiome_col]]], n_taxa)[0],
                dtype=torch.float
            )
        else:
            micro = torch.zeros(n_taxa)
        records.append({
            "drug_graph":     g,
            "microbiome":     micro,
            "bioavailability": float(row.get(label_col, 0.5)),
            "response":       int(row.get("response_class", 1)),
        })

    out = PROC / "dataset.pkl"
    with open(out, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} records → {out}")
    return records


# ── Synthetic dataset for CI/quick-start ──────────────────────────
SAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # Ibuprofen
    "O=C(O)c1ccccc1O",  # Salicylic acid
]

def make_synthetic(n_samples=500, n_taxa=1000, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_samples):
        smi = SAMPLE_SMILES[i % len(SAMPLE_SMILES)]
        g   = smiles_to_graph(smi)
        if g is None:
            continue
        micro  = torch.tensor(rng.exponential(1.0, n_taxa).astype(np.float32))
        bioa   = float(rng.uniform(0.2, 0.9))
        resp   = int(rng.integers(0, 3))
        records.append({"drug_graph": g, "microbiome": micro,
                        "bioavailability": bioa, "response": resp})
    out = PROC / "synthetic_dataset.pkl"
    with open(out, "wb") as f:
        pickle.dump(records, f)
    print(f"Synthetic: {len(records)} records → {out}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Build synthetic dataset for testing")
    parser.add_argument("--interactions", type=str, help="Path to interaction CSV")
    parser.add_argument("--otu",          type=str, help="Path to OTU abundance table")
    parser.add_argument("--n-taxa",       type=int, default=1000)
    args = parser.parse_args()

    if args.synthetic:
        make_synthetic(n_taxa=args.n_taxa)
    elif args.interactions:
        build_dataset(args.interactions, otu_table=args.otu, n_taxa=args.n_taxa)
    else:
        print("Use --synthetic for quick-start, or --interactions <csv> for real data")
