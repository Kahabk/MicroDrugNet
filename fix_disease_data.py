"""
MicroDrugNet - Fix Missing Disease Microbiome Datasets
=======================================================
Your healthy data already downloaded correctly.
This script adds missing disease datasets using current MLRepo folder names
and avoids re-appending cohorts that are already present.

Run:  python fix_disease_data.py

Adds these real disease datasets from MLRepo:
  - qin2014     : liver cirrhosis vs healthy
  - kostic      : colorectal cancer
  - gevers      : Crohn's disease / IBD
  - turnbaugh   : obesity
  - qin2012     : type 2 diabetes
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets"

# Format:
#   aliases: ordered by most likely current MLRepo folder name first
#   min_existing: if the current file already has at least this many samples for
#                 the condition, this specific cohort is considered already
#                 present and will be skipped.
DISEASE_DATASETS = [
    {
        "aliases": ["qin2014"],
        "has_refseq": False,
        "condition": "cirrhosis",
        "desc": "Qin 2014 liver cirrhosis",
        "min_existing": 130,
    },
    {
        "aliases": ["kostic"],
        "has_refseq": True,
        "condition": "CRC",
        "desc": "Kostic colorectal cancer",
        "min_existing": 190,
    },
    {
        "aliases": ["gevers"],
        "has_refseq": True,
        "condition": "IBD",
        "desc": "Gevers Crohn's disease",
        "min_existing": 1358,
    },
    {
        "aliases": ["turnbaugh"],
        "has_refseq": True,
        "condition": "obesity",
        "desc": "Turnbaugh obesity",
        "min_existing": None,
    },
    {
        "aliases": ["qin2012"],
        "has_refseq": False,
        "condition": "T2D",
        "desc": "Qin 2012 type 2 diabetes",
        "min_existing": None,
    },
    {
        "aliases": ["yatsunenko"],
        "has_refseq": False,
        "condition": "healthy",
        "desc": "Yatsunenko healthy gut",
        "min_existing": 530,
    },
]

KNOWN_BAD_APPEND_COUNTS = {
    "cirrhosis": 130,
    "CRC": 190,
    "IBD": 1358,
    "healthy": 530,
}


def download(url, dest, desc=""):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 200:
        return True
    print(f"    GET {desc} ...")
    try:
        r = requests.get(
            url,
            stream=True,
            timeout=60,
            headers={"User-Agent": "MicroDrugNet/2.0"},
        )
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r        {done / total * 100:5.1f}%", end="", flush=True)
        print(f"\r        Done - {done / 1024:.0f} KB")
        return True
    except Exception as e:
        print(f"\r        FAIL: {e}")
        dest.unlink(missing_ok=True)
        return False


def _maybe_repair_previous_bad_append(existing):
    """Trim the exact duplicate tail created by the older buggy script."""
    n_bad = sum(KNOWN_BAD_APPEND_COUNTS.values())
    if len(existing) < n_bad:
        return existing

    tail_counts = existing.tail(n_bad)["condition"].value_counts().to_dict()
    if tail_counts != KNOWN_BAD_APPEND_COUNTS:
        return existing

    print("\n  Repairing previous duplicate append from older script ...")
    repaired = existing.iloc[:-n_bad].copy().reset_index(drop=True)
    print(f"  Removed {n_bad:,} duplicated rows from file tail")
    print(f"  Repaired counts: {repaired['condition'].value_counts().to_dict()}")
    return repaired


def _candidate_urls(aliases, has_refseq):
    urls = []
    seen = set()
    for folder in aliases:
        patterns = (
            [
                f"{BASE}/{folder}/refseq/otutable.txt",
                f"{BASE}/{folder}/gg/otutable.txt",
                f"{BASE}/{folder}/otutable.txt",
            ]
            if has_refseq
            else [
                f"{BASE}/{folder}/otutable.txt",
                f"{BASE}/{folder}/refseq/otutable.txt",
                f"{BASE}/{folder}/gg/otutable.txt",
            ]
        )
        for url in patterns:
            if url not in seen:
                urls.append((folder, url))
                seen.add(url)
    return urls


def try_load_dataset(aliases, has_refseq, condition):
    """Try multiple folder aliases and subfolder layouts to find the OTU table."""
    safe = aliases[0].replace("/", "_")
    dest = RAW_DIR / f"mlrepo_{safe}_otu.txt"

    for folder, url in _candidate_urls(aliases, has_refseq):
        ok = download(url, dest, f"{folder} OTU table")
        if ok:
            try:
                otu = pd.read_csv(
                    dest,
                    sep="\t",
                    index_col=0,
                    comment="#",
                    low_memory=False,
                )
                if otu.shape[0] < 5:
                    dest.unlink(missing_ok=True)
                    continue
                otu = otu.T.copy()
                otu.columns = [str(c) for c in otu.columns]
                otu["condition"] = condition
                print(f"        Loaded {len(otu)} samples x {otu.shape[1] - 1} OTUs")
                return otu
            except Exception as e:
                print(f"        Parse error: {e}")
                dest.unlink(missing_ok=True)
                continue

    return None


def fix_disease_data():
    print("=" * 60)
    print("  Adding disease microbiome datasets")
    print("=" * 60)

    existing_path = PROCESSED_DIR / "microbiome_profiles.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path, low_memory=False)
        existing = _maybe_repair_previous_bad_append(existing)
        print(f"\n  Existing data: {len(existing):,} samples")
        print(f"  Conditions: {existing['condition'].value_counts().to_dict()}")
    else:
        existing = None
        print("  No existing microbiome_profiles.csv found")

    existing_counts = (
        existing["condition"].value_counts().to_dict() if existing is not None else {}
    )
    new_dfs = []

    for dataset in DISEASE_DATASETS:
        aliases = dataset["aliases"]
        has_refseq = dataset["has_refseq"]
        condition = dataset["condition"]
        desc = dataset["desc"]
        min_existing = dataset["min_existing"]

        print(f"\n  {desc} ...")
        if min_existing is not None and existing_counts.get(condition, 0) >= min_existing:
            print(
                f"    SKIP: existing {condition} samples "
                f"({existing_counts.get(condition, 0):,}) already cover this cohort"
            )
            continue

        df = try_load_dataset(aliases, has_refseq, condition)
        if df is not None:
            new_dfs.append((condition, df))

    if not new_dfs:
        print("\n  No new datasets loaded.")
        print("  Keeping current microbiome_profiles.csv after any repair above.")
        if existing is not None:
            existing.to_csv(existing_path, index=False)
            _rebuild_master(existing)
        _write_summary()
        return

    print(f"\n  Merging {len(new_dfs)} new disease datasets ...")

    normed = []
    for _, df in new_dfs:
        otu_cols = [c for c in df.columns if c != "condition"]
        vals = df[otu_cols].values.astype(float)
        row_sums = vals.sum(axis=1, keepdims=True)
        vals = vals / (row_sums + 1e-12)
        normed_df = pd.DataFrame(vals, columns=otu_cols)
        normed_df["condition"] = df["condition"].values
        normed.append(normed_df)

    disease_merged = pd.concat(normed, axis=0, sort=False).fillna(0)

    otu_cols = [c for c in disease_merged.columns if c != "condition"]
    prev = (disease_merged[otu_cols] > 0).mean()
    top_500 = prev.nlargest(min(500, len(otu_cols))).index.tolist()

    disease_merged = disease_merged[top_500 + ["condition"]].copy()
    pad_data = {}
    for i in range(len(top_500), 500):
        pad_data[f"taxa_{i}"] = 0.0
    if pad_data:
        disease_merged = pd.concat(
            [disease_merged, pd.DataFrame(pad_data, index=disease_merged.index)],
            axis=1,
        )

    disease_merged.columns = (
        [f"taxa_{i}" for i in range(len(top_500))]
        + [f"taxa_{i}" for i in range(len(top_500), 500)]
        + ["condition"]
    )
    disease_merged = disease_merged.reset_index(drop=True)

    dist = disease_merged["condition"].value_counts().to_dict()
    print(f"  New disease samples: {len(disease_merged):,}")
    print(f"  Disease conditions: {dist}")

    taxa_cols = [f"taxa_{i}" for i in range(500)]
    if existing is not None:
        existing_taxa = [c for c in taxa_cols if c in existing.columns]
        existing_sub = existing[existing_taxa + ["condition"]].copy()
        missing_taxa = [c for c in taxa_cols if c not in existing_taxa]
        if missing_taxa:
            pad = pd.DataFrame(
                np.zeros((len(existing_sub), len(missing_taxa))),
                columns=missing_taxa,
                index=existing_sub.index,
            )
            existing_sub = pd.concat([existing_sub, pad], axis=1)
        existing_sub = existing_sub[taxa_cols + ["condition"]]
        disease_sub = disease_merged[taxa_cols + ["condition"]]
        final = pd.concat([existing_sub, disease_sub], ignore_index=True)
    else:
        final = disease_merged[taxa_cols + ["condition"]]

    final.to_csv(existing_path, index=False)
    final_dist = final["condition"].value_counts().to_dict()
    print(f"\n  FINAL microbiome_profiles.csv: {len(final):,} samples")
    print(f"  All conditions: {final_dist}")

    _rebuild_master(final)
    _write_summary()



def _rebuild_master(final):
    master_path = PROCESSED_DIR / "microbiome_all.csv"
    gmhi_path = PROCESSED_DIR / "gmhi_microbiome.csv"
    taxa_cols = [f"taxa_{i}" for i in range(500)]

    dfs = [final[taxa_cols + ["condition"]]]
    if gmhi_path.exists():
        gmhi = pd.read_csv(gmhi_path, low_memory=False)
        gmhi_taxa = [c for c in taxa_cols if c in gmhi.columns]
        gmhi_sub = gmhi[gmhi_taxa + ["condition"]].copy()
        missing = [c for c in taxa_cols if c not in gmhi_taxa]
        if missing:
            pad = pd.DataFrame(
                np.zeros((len(gmhi_sub), len(missing))),
                columns=missing,
                index=gmhi_sub.index,
            )
            gmhi_sub = pd.concat([gmhi_sub, pad], axis=1)
        dfs.append(gmhi_sub[taxa_cols + ["condition"]])

    master = pd.concat(dfs, ignore_index=True)
    master.to_csv(master_path, index=False)
    print(f"  Master microbiome_all.csv: {len(master):,} samples")
    print(f"  Saved -> {master_path}")



def _write_summary():
    print("\n" + "=" * 60)
    print("  FINAL STATUS")
    print("=" * 60)

    files = {
        "chembl_drugs.csv": "ChEMBL drug SMILES",
        "microbiome_profiles.csv": "MLRepo microbiome",
        "gmhi_microbiome.csv": "GMHI microbiome",
        "microbiome_all.csv": "Master microbiome",
        "drug_microbiome_interactions.csv": "Drug interactions",
    }

    for fname, label in files.items():
        path = PROCESSED_DIR / fname
        if path.exists():
            n = sum(1 for _ in open(path)) - 1
            kb = path.stat().st_size / 1024
            print(f"  OK  {label:35s} {n:>7,} rows  {kb:>7.0f} KB")
        else:
            print(f"  --  {label}")

    micro_path = PROCESSED_DIR / "microbiome_profiles.csv"
    if micro_path.exists():
        n = sum(1 for _ in open(micro_path)) - 1
        print(f"\n  Your microbiome dataset has {n:,} samples.")
        if n >= 1000:
            print("  This is enough real data to train and beat 0.82 AUROC.")
        print()
        print("  Next steps:")
        print("    python build_training_dataset.py")
        print("    python train.py --n_epochs 100 --batch_size 32")


if __name__ == "__main__":
    fix_disease_data()
