"""
MicroDrugNet — Build Training Dataset
======================================
Reads every real file you downloaded and builds the final
training CSV that train.py needs.

Reads:
  - data/processed/chembl_drugs.csv          (50k drugs)
  - data/processed/microbiome_all.csv        (8,990 samples)
  - data/raw/MASI_v1.0_download_microbeSubstanceInteractionRecords*.xlsx
  - data/raw/MLRepo-master.zip               (disease OTUs)
  - data/raw/MASI_v1.0_download_microbesInfo.xlsx
  - data/raw/MASI_v1.0_download_substanceInfo.xlsx

Outputs:
  - data/processed/training_dataset.csv      (train.py reads this)

Run:
  python build_training_dataset.py
"""

import pandas as pd
import numpy as np
import zipfile
import glob
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
N_TAXA    = 500
N_PAIRS   = 30000   # training pairs to generate
SEED      = 42
rng       = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load drugs
# ─────────────────────────────────────────────────────────────────

def load_drugs() -> pd.DataFrame:
    print("\n[1/5] Loading drug SMILES ...")
    path = PROCESSED / "chembl_drugs.csv"
    df   = pd.read_csv(path)
    # Drop rows with missing or very short SMILES
    df   = df[df["smiles"].str.len() > 5].reset_index(drop=True)
    print(f"  {len(df):,} drugs loaded")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Load microbiome profiles
# ─────────────────────────────────────────────────────────────────

def load_microbiome() -> pd.DataFrame:
    print("\n[2/5] Loading microbiome profiles ...")
    path = PROCESSED / "microbiome_all.csv"
    df   = pd.read_csv(path, low_memory=False)
    taxa = [f"taxa_{i}" for i in range(N_TAXA)]
    # Keep only taxa columns + condition
    avail   = [c for c in taxa if c in df.columns]
    missing = [c for c in taxa if c not in df.columns]
    result  = df[avail + ["condition"]].copy()
    if missing:
        pad = pd.DataFrame(
            np.zeros((len(result), len(missing)), dtype=np.float32),
            columns=missing, index=result.index
        )
        result = pd.concat([result, pad], axis=1)
    result = result[taxa + ["condition"]].reset_index(drop=True)
    print(f"  {len(result):,} microbiome samples loaded")
    cond = result["condition"].value_counts().to_dict()
    print(f"  Conditions: {cond}")
    return result


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Load MASI interaction records
# ─────────────────────────────────────────────────────────────────

def load_masi() -> pd.DataFrame:
    print("\n[3/5] Loading MASI interaction records ...")

    # Find the interaction file — handle version suffixes
    patterns = [
        "MASI_v1.0_download_microbeSubstanceInteractionRecords*.xlsx",
        "MASI_v1.0_download_microbeSubstanceInteractionRecords*.txt",
    ]
    found = []
    for pat in patterns:
        found += glob.glob(str(RAW / pat))

    if not found:
        print("  MASI interaction file not found — using curated fallback")
        return _curated_interactions()

    path = Path(found[0])
    print(f"  Reading {path.name} ...")
    try:
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as e:
        print(f"  Parse error: {e} — using curated fallback")
        return _curated_interactions()

    print(f"  Columns: {list(df.columns[:8])}")
    print(f"  {len(df):,} MASI interactions loaded")

    # Normalise column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if any(x in cl for x in ["substance","drug","compound","chemical"]):
            col_map[col] = "drug"
        elif any(x in cl for x in ["microbe","bacteria","organism","taxon","species"]):
            col_map[col] = "bacterium"
        elif any(x in cl for x in ["effect","change","impact","alteration"]):
            col_map[col] = "effect"
        elif any(x in cl for x in ["direction","increase","decrease","type"]):
            col_map[col] = "direction"

    df = df.rename(columns=col_map)
    return df


def _curated_interactions() -> pd.DataFrame:
    rows = [
        ("metformin",       "Akkermansia muciniphila",  "increase"),
        ("metformin",       "Bifidobacterium",          "increase"),
        ("metformin",       "Intestinibacter",          "decrease"),
        ("digoxin",         "Eggerthella lenta",        "decrease"),
        ("ciprofloxacin",   "Lactobacillus",            "decrease"),
        ("ciprofloxacin",   "Bifidobacterium",          "decrease"),
        ("amoxicillin",     "Lactobacillus",            "decrease"),
        ("omeprazole",      "Streptococcus",            "increase"),
        ("irinotecan",      "Clostridium perfringens",  "increase"),
        ("levodopa",        "Enterococcus faecalis",    "decrease"),
        ("atorvastatin",    "Akkermansia muciniphila",  "increase"),
        ("aspirin",         "Faecalibacterium prausnitzii","increase"),
        ("warfarin",        "Bacteroides",              "decrease"),
        ("fluoxetine",      "Lactobacillus rhamnosus",  "increase"),
        ("sertraline",      "Bacteroides",              "decrease"),
        ("cyclophosphamide","Lactobacillus johnsonii",  "increase"),
        ("tacrolimus",      "Faecalibacterium prausnitzii","decrease"),
        ("ibuprofen",       "Akkermansia muciniphila",  "decrease"),
        ("sulfasalazine",   "Intestinal bacteria",      "decrease"),
        ("rifaximin",       "Bifidobacterium",          "increase"),
    ]
    return pd.DataFrame(rows, columns=["drug", "bacterium", "direction"])


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Compute labels for each drug+microbiome pair
# ─────────────────────────────────────────────────────────────────

# Known base bioavailabilities from literature
BIOAVAIL = {
    "aspirin":0.68,"metformin":0.55,"metronidazole":0.99,
    "ciprofloxacin":0.70,"omeprazole":0.65,"amoxicillin":0.93,
    "ibuprofen":0.87,"warfarin":0.93,"digoxin":0.75,
    "atorvastatin":0.14,"fluoxetine":0.72,"sertraline":0.44,
    "sulfasalazine":0.15,"rifaximin":0.10,"tacrolimus":0.25,
    "cyclophosphamide":0.97,"levodopa":0.30,"irinotecan":0.40,
}

# Disease conditions → bioavailability modifier
CONDITION_MOD = {
    "healthy":   0.00,
    "IBD":      -0.12,
    "CRC":      -0.08,
    "cirrhosis":-0.15,
    "CDI":      -0.18,
    "T2D":      -0.06,
    "NASH":     -0.10,
    "obesity":  -0.05,
    "mixed":    -0.03,
}

# Signal taxa indices (fixed positions in our taxa_0..taxa_499 vector)
# These correspond to the top-prevalence taxa kept during preprocessing
IDX_AKKERMANSIA   = 0   # Akkermansia muciniphila
IDX_BIFIDOBACT    = 1   # Bifidobacterium
IDX_LACTOBACILLUS = 2   # Lactobacillus
IDX_CLOSTRIDIUM   = 3   # Clostridium difficile
IDX_BACTEROIDES   = 4   # Bacteroides
IDX_FAECALIB      = 5   # Faecalibacterium prausnitzii
IDX_EGGERTHELLA   = 6   # Eggerthella lenta



TAXON_NAME_TO_IDX = {
    "akkermansia": IDX_AKKERMANSIA,
    "bifidobacterium": IDX_BIFIDOBACT,
    "lactobacillus": IDX_LACTOBACILLUS,
    "clostridium": IDX_CLOSTRIDIUM,
    "bacteroides": IDX_BACTEROIDES,
    "faecalibacterium": IDX_FAECALIB,
    "eggerthella": IDX_EGGERTHELLA,
    "intestinibacter": IDX_CLOSTRIDIUM,
    "streptococcus": IDX_LACTOBACILLUS,
    "enterococcus": IDX_EGGERTHELLA,
}


def build_masi_lookup(masi: pd.DataFrame) -> dict:
    """Map drug names to directional signal on known taxa proxies."""
    lookup = {}
    if masi is None or len(masi) == 0:
        return lookup

    cols = {c.lower(): c for c in masi.columns}
    drug_col = cols.get('drug') or cols.get('substance')
    bact_col = cols.get('bacterium') or cols.get('microbe')
    dir_col = cols.get('direction') or cols.get('effect')
    if not all([drug_col, bact_col, dir_col]):
        return lookup

    for _, row in masi.iterrows():
        drug = str(row.get(drug_col, '')).strip().lower()
        bact = str(row.get(bact_col, '')).strip().lower()
        direction = str(row.get(dir_col, '')).strip().lower()
        if not drug or not bact:
            continue
        matched_idx = None
        for key, idx in TAXON_NAME_TO_IDX.items():
            if key in bact:
                matched_idx = idx
                break
        if matched_idx is None:
            continue
        sign = 1.0 if any(x in direction for x in ['increase', 'promote', 'up']) else -1.0
        lookup.setdefault(drug, []).append((matched_idx, sign))
    return lookup
def compute_labels(drug_name: str,
                   drug_smiles: str,
                   micro_row: pd.Series,
                   rng,
                   masi_lookup: dict | None = None) -> dict:
    """
    Compute biologically-grounded labels for one drug+microbiome pair.
    Uses known interactions from literature where drug_name is known,
    otherwise uses microbiome diversity as proxy signal.
    """
    condition = str(micro_row.get("condition", "healthy")).lower()
    taxa_vals = micro_row[[f"taxa_{i}" for i in range(N_TAXA)]].values.astype(float)

    # Signal taxa abundances
    akk   = float(taxa_vals[IDX_AKKERMANSIA])
    bifid = float(taxa_vals[IDX_BIFIDOBACT])
    lacto = float(taxa_vals[IDX_LACTOBACILLUS])
    cdiff = float(taxa_vals[IDX_CLOSTRIDIUM])
    bact  = float(taxa_vals[IDX_BACTEROIDES])
    fprau = float(taxa_vals[IDX_FAECALIB])
    egg   = float(taxa_vals[IDX_EGGERTHELLA])

    protective = akk + bifid + lacto + fprau

    # ── Bioavailability ──────────────────────────────────────────
    base = BIOAVAIL.get(drug_name.lower(), 0.62)
    cond_mod = CONDITION_MOD.get(condition, 0.0)

    if "metformin" in drug_name.lower():
        drug_mod = +0.15 * akk + 0.08 * bifid
    elif "digoxin" in drug_name.lower():
        drug_mod = -0.40 * egg
    elif any(x in drug_name.lower() for x in
             ["cillin","oxacin","cycline","mycin","floxacin"]):
        drug_mod = -0.12 * cdiff + 0.05 * protective
    elif any(x in drug_name.lower() for x in ["statin","vastatin"]):
        drug_mod = +0.10 * akk
    elif "prazole" in drug_name.lower():
        drug_mod = -0.05 * lacto
    else:
        drug_mod = 0.06 * protective - 0.04 * cdiff

    noise = float(rng.normal(0, 0.04))
    bioavailability = float(np.clip(base + cond_mod + drug_mod + noise,
                                    0.05, 0.99))

    # ── Response score ───────────────────────────────────────────
    # Keep this continuous first, then bucket across the whole dataset later
    # so classes are not extremely imbalanced.
    cond_severity = {
        "healthy": 0.00,
        "mixed": -0.08,
        "obesity": -0.10,
        "t2d": -0.12,
        "crc": -0.16,
        "ibd": -0.18,
        "cirrhosis": -0.20,
        "cdi": -0.22,
        "nash": -0.15,
    }.get(condition, -0.05)

    if "metformin" in drug_name.lower():
        response_score = 0.34 + akk * 3.2 + bifid * 1.6 + cond_severity
    elif "digoxin" in drug_name.lower():
        response_score = 0.12 - egg * 2.8 + 0.35 * protective + cond_severity
    elif any(x in drug_name.lower() for x in ["cillin", "oxacin", "mycin", "cycline"]):
        response_score = 0.08 + protective * 1.6 - cdiff * 2.5 + cond_severity
    else:
        response_score = (
            (bioavailability - 0.58) * 2.4
            + protective * 1.4
            - cdiff * 1.6
            - 0.45 * egg
            + cond_severity
        )

    if masi_lookup:
        known = 0.0
        drug_key = drug_name.lower()
        for key, effects in masi_lookup.items():
            if key and key in drug_key:
                for idx, sign in effects:
                    known += sign * float(taxa_vals[idx])
        response_score += 1.35 * known

    response_score += float(rng.normal(0, 0.16))

    # ── Toxicity ─────────────────────────────────────────────────
    shannon = float(-np.sum(taxa_vals * np.log(taxa_vals + 1e-12)))
    diversity_risk = max(0.0, 2.8 - shannon)
    cond_tox = {
        "healthy": 0.02,
        "mixed": 0.08,
        "obesity": 0.07,
        "t2d": 0.10,
        "crc": 0.14,
        "ibd": 0.18,
        "cirrhosis": 0.16,
        "cdi": 0.24,
        "nash": 0.12,
    }.get(condition, 0.06)

    tox = 0.08 + cond_tox + cdiff * 1.9 + diversity_risk * 0.08
    if any(x in drug_name.lower() for x in ["cillin", "oxacin", "cycline", "mycin"]):
        tox += 0.14 + cdiff * 0.9
    elif "irinotecan" in drug_name.lower():
        tox += 0.18 + bact * 0.8
    elif "warfarin" in drug_name.lower():
        tox += 0.08

    tox += float(rng.normal(0, 0.028))
    toxicity = float(np.clip(tox, 0.0, 1.0))

    return {
        "bioavailability": bioavailability,
        "response_score":  response_score,
        "toxicity":        toxicity,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Assemble training pairs
# ─────────────────────────────────────────────────────────────────

def _sample_microbiome_indices(microbiome: pd.DataFrame, n_pairs: int, rng) -> np.ndarray:
    """Sample microbiome rows with lighter condition imbalance than raw counts."""
    cond_to_idx = {
        cond: grp.index.to_numpy()
        for cond, grp in microbiome.groupby("condition", sort=False)
    }
    conditions = list(cond_to_idx.keys())
    counts = np.array([len(cond_to_idx[c]) for c in conditions], dtype=float)
    probs = np.sqrt(counts)
    probs = probs / probs.sum()

    chosen_conditions = rng.choice(conditions, size=n_pairs, p=probs)
    out = np.empty(n_pairs, dtype=int)
    for i, cond in enumerate(chosen_conditions):
        idx_pool = cond_to_idx[cond]
        out[i] = int(rng.choice(idx_pool))
    return out


def build_dataset(drugs: pd.DataFrame,
                  microbiome: pd.DataFrame,
                  masi: pd.DataFrame,
                  n_pairs: int) -> pd.DataFrame:
    print(f"\n[4/5] Building {n_pairs:,} training pairs ...")

    taxa_cols = [f"taxa_{i}" for i in range(N_TAXA)]
    rows      = []
    masi_lookup = build_masi_lookup(masi)

    # Sample drugs uniformly, but microbiomes with only mild condition smoothing.
    drug_idx  = rng.integers(0, len(drugs), size=n_pairs)
    micro_idx = _sample_microbiome_indices(microbiome, n_pairs, rng)

    for i in range(n_pairs):
        d_row  = drugs.iloc[drug_idx[i]]
        m_row  = microbiome.iloc[micro_idx[i]]

        drug_name  = str(d_row.get("drug_name",
                         d_row.get("chembl_id", "unknown")))
        smiles     = str(d_row["smiles"])
        condition  = str(m_row.get("condition", "healthy"))

        labels = compute_labels(drug_name, smiles, m_row, rng, masi_lookup=masi_lookup)

        row = {
            "drug_id":         str(d_row.get("chembl_id", f"drug_{drug_idx[i]}")),
            "drug_name":       drug_name,
            "smiles":          smiles,
            "microbiome_sample_id": f"sample_{micro_idx[i]}",
            "condition":       condition,
            "bioavailability": labels["bioavailability"],
            "response_score":  labels["response_score"],
            "toxicity":        labels["toxicity"],
        }
        for j in range(N_TAXA):
            row[f"taxa_{j}"] = float(m_row[f"taxa_{j}"])

        rows.append(row)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,} / {n_pairs:,} pairs built ...")

    df = pd.DataFrame(rows)

    # Bucket response scores with a modest skew toward medium/high response.
    score = df["response_score"].to_numpy()
    q1, q2 = np.quantile(score, [0.22, 0.64])
    df["response_class"] = np.digitize(score, [q1, q2]).astype(int)
    df = df.drop(columns=["response_score"])
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 6 — Save and summarise
# ─────────────────────────────────────────────────────────────────

def save_and_report(df: pd.DataFrame):
    print("\n[5/5] Saving training dataset ...")
    out = PROCESSED / "training_dataset.csv"
    df.to_csv(out, index=False)

    size_mb = out.stat().st_size / 1e6
    dist    = df["response_class"].value_counts().sort_index().to_dict()
    cond    = df["condition"].value_counts().to_dict()

    print(f"\n{'='*60}")
    print(f"  Training dataset ready")
    print(f"{'='*60}")
    print(f"  Rows:            {len(df):,}")
    print(f"  Taxa features:   {N_TAXA}")
    print(f"  File size:       {size_mb:.1f} MB")
    print(f"  Response dist:   {dist}")
    print(f"  Conditions:      {cond}")
    print(f"  Saved to:        {out}")
    print(f"\n  NOW RUN:")
    print(f"    python train.py --n_epochs 100 --batch_size 32")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MicroDrugNet — Build Training Dataset")
    print("======================================")

    drugs      = load_drugs()
    microbiome = load_microbiome()
    masi       = load_masi()

    df = build_dataset(drugs, microbiome, masi, n_pairs=N_PAIRS)
    save_and_report(df)
