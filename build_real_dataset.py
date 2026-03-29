"""
MicroDrugNet — Rebuild Training Dataset with Real MASI Labels
=============================================================
Uses MASI's real experimentally measured outcomes as labels.

Key columns used:
  Substance-Name          -> drug name (match to ChEMBL SMILES)
  Microbe-Name            -> bacterium
  Microbe_Change          -> increase/decrease/no change
  Metabolism_Effect_on_Drug -> Increase Efficacy / Decrease Efficacy /
                               Increase Toxicity / Decrease Toxicity
  Outcome                 -> free text describing the result
  Model_Condition/Disease -> disease context

Run:
    python build_real_dataset.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
N_TAXA    = 500
N_PAIRS   = 30000
SEED      = 42
rng       = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────
# 1. Load and parse MASI
# ─────────────────────────────────────────────────────────────────

def load_masi_real():
    print("\n[1/6] Loading real MASI interaction data ...")
    path = RAW / "MASI_v1.0_download_microbeSubstanceInteractionRecords_ver20200928.xlsx"
    df   = pd.read_excel(path)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ── Parse response class from Metabolism_Effect_on_Drug ──────
    def parse_effect(val):
        if pd.isna(val):
            return None
        v = str(val).lower()
        if "increase efficacy" in v:
            return 2          # high response
        elif "decrease efficacy" in v:
            return 0          # low response
        elif "increase toxicity" in v:
            return -1         # toxic (separate label)
        elif "decrease toxicity" in v:
            return 1          # medium / neutral
        return None

    # ── Parse bioavailability direction from Outcome ──────────────
    def parse_bioavail_dir(val):
        if pd.isna(val):
            return 0.0
        v = str(val).lower()
        if any(x in v for x in ["increased absorption", "increased bioavail",
                                  "increased activity", "increase systemic"]):
            return +0.15
        elif any(x in v for x in ["decreased absorption", "decreased bioavail",
                                   "decreased activity", "decrease systemic",
                                   "decreased intestinal"]):
            return -0.15
        elif "no change" in v or "no significant" in v:
            return 0.0
        return 0.0

    # ── Parse toxicity from Outcome ───────────────────────────────
    def parse_toxicity(effect_val, outcome_val):
        e = str(effect_val).lower() if not pd.isna(effect_val) else ""
        o = str(outcome_val).lower() if not pd.isna(outcome_val) else ""
        combined = e + " " + o
        if any(x in combined for x in ["toxicity", "side effect", "toxic",
                                         "adverse", "nausea", "diarrhea",
                                         "colitis"]):
            if "increase toxicity" in combined:
                return 0.75
            elif "decrease toxicity" in combined:
                return 0.10
            else:
                return 0.45
        return 0.15

    # ── Parse microbe change ──────────────────────────────────────
    def parse_microbe_change(val):
        if pd.isna(val):
            return 0.0
        v = str(val).lower()
        if "increase" in v or "enrich" in v or "promot" in v:
            return +1.0
        elif "decrease" in v or "reduc" in v or "inhibit" in v or "deplet" in v:
            return -1.0
        return 0.0

    df["_effect"]         = df["Metabolism_Effect_on_Drug"].apply(parse_effect)
    df["_bioavail_delta"] = df["Outcome"].apply(parse_bioavail_dir)
    df["_toxicity"]       = df.apply(
        lambda r: parse_toxicity(r["Metabolism_Effect_on_Drug"], r["Outcome"]),
        axis=1
    )
    df["_microbe_change"] = df["Microbe_Change"].apply(parse_microbe_change)
    df["_drug_name_clean"] = df["Substance-Name"].str.lower().str.strip()
    df["_microbe_clean"]   = df["Microbe-Name"].str.lower().str.strip()

    # Keep only rows with a parseable effect
    df_valid = df[df["_effect"].notna()].copy()
    print(f"  {len(df_valid):,} rows with parseable effect labels")

    effect_dist = df_valid["_effect"].value_counts().to_dict()
    print(f"  Effect distribution: {effect_dist}")

    return df, df_valid


# ─────────────────────────────────────────────────────────────────
# 2. Load MASI substance info to get SMILES via ChEMBL match
# ─────────────────────────────────────────────────────────────────

def load_masi_substance_info():
    print("\n[2/6] Loading MASI substance info ...")
    path = RAW / "MASI_v1.0_download_substanceInfo.xlsx"
    if not path.exists():
        print("  Not found — will match by name only")
        return pd.DataFrame()
    df = pd.read_excel(path)
    print(f"  {len(df)} substances, columns: {list(df.columns[:6])}")
    return df


# ─────────────────────────────────────────────────────────────────
# 3. Load ChEMBL drugs
# ─────────────────────────────────────────────────────────────────

def load_chembl():
    print("\n[3/6] Loading drug catalog ...")
    chembl = pd.read_csv(PROCESSED / "chembl_drugs.csv")
    chembl = chembl[chembl["smiles"].str.len() > 5].reset_index(drop=True)
    chembl["drug_name"] = chembl["chembl_id"].astype(str).str.lower()
    chembl["_name_clean"] = chembl["drug_name"]
    chembl["_source"] = "chembl"

    catalogs = [chembl[["chembl_id", "drug_name", "smiles", "_name_clean", "_source"]]]

    curated = pd.DataFrame(
        {
            "chembl_id": [f"curated_{name}" for name in CURATED_DRUG_SMILES],
            "drug_name": list(CURATED_DRUG_SMILES.keys()),
            "smiles": list(CURATED_DRUG_SMILES.values()),
        }
    )
    curated["_name_clean"] = curated["drug_name"].str.lower().str.strip()
    curated["_source"] = "curated"
    catalogs.insert(0, curated[["chembl_id", "drug_name", "smiles", "_name_clean", "_source"]])

    magmd_path = PROCESSED / "magmd_drug_metabolism.csv"
    if magmd_path.exists():
        magmd = pd.read_csv(magmd_path)
        magmd = magmd[magmd["smiles"].astype(str).str.len() > 5].copy()
        magmd["drug_name"] = magmd["drug_name"].astype(str)
        magmd["_name_clean"] = magmd["drug_name"].str.lower().str.strip()
        magmd["chembl_id"] = magmd["_name_clean"]
        magmd["_source"] = "magmd"
        catalogs.insert(0, magmd[["chembl_id", "drug_name", "smiles", "_name_clean", "_source"]])

    df = pd.concat(catalogs, ignore_index=True, sort=False)
    df = df.drop_duplicates(subset=["_name_clean", "smiles"], keep="first").reset_index(drop=True)
    named = int((df["_source"] != "chembl").sum())
    print(f"  {len(df):,} total compounds")
    print(f"  Named local compounds: {named:,}")
    return df


# ─────────────────────────────────────────────────────────────────
# 4. Load microbiome profiles
# ─────────────────────────────────────────────────────────────────

def load_microbiome():
    print("\n[4/6] Loading microbiome profiles ...")
    df = pd.read_csv(PROCESSED / "microbiome_all.csv", low_memory=False)
    taxa = [f"taxa_{i}" for i in range(N_TAXA)]
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
    result["microbiome_sample_id"] = [f"sample_{i}" for i in range(len(result))]
    print(f"  {len(result):,} samples")
    print(f"  Conditions: {result['condition'].value_counts().to_dict()}")
    return result


# ─────────────────────────────────────────────────────────────────
# 5. Build training pairs
#    Strategy:
#    - For MASI-matched drugs: use real MASI labels directly
#    - For unmatched drugs: use biology-based formula (as before)
#    This gives a mixture of real + derived labels
# ─────────────────────────────────────────────────────────────────

# Base bioavailabilities from literature
BIOAVAIL = {
    "aspirin":0.68,"metformin":0.55,"metronidazole":0.99,
    "ciprofloxacin":0.70,"omeprazole":0.65,"amoxicillin":0.93,
    "ibuprofen":0.87,"warfarin":0.93,"digoxin":0.75,
    "atorvastatin":0.14,"fluoxetine":0.72,"sertraline":0.44,
    "sulfasalazine":0.15,"rifaximin":0.10,"tacrolimus":0.25,
    "cyclophosphamide":0.97,"levodopa":0.30,"irinotecan":0.40,
    "baicalin":0.30,"olsalazine":0.20,"nizatidine":0.70,
    "sorivudine":0.80,"prontosil":0.85,"gemcitabine":0.90,
    "irinotecan":0.40,"oxaliplatin":0.95,"5-fluorouracil":0.74,
}

CONDITION_MOD = {
    "healthy":0.00,"IBD":-0.12,"CRC":-0.08,
    "cirrhosis":-0.15,"CDI":-0.18,"T2D":-0.06,
    "NASH":-0.10,"obesity":-0.05,"mixed":-0.03,
}

MICROBE_TO_TAXA_IDX = {
    "akkermansia": 0,
    "bifidobacterium": 1,
    "lactobacillus": 2,
    "clostridium": 3,
    "clostridioides": 3,
    "bacteroides": 4,
    "faecalibacterium": 5,
    "eggerthella": 6,
    "intestinibacter": 3,
    "streptococcus": 2,
    "enterococcus": 6,
}

MANUAL_DRUG_ALIASES = {
    "5-aminosalicylic acid": "mesalazine",
    "acetylsalicylic acid": "aspirin",
    "mesalamine": "mesalazine",
}

CURATED_DRUG_SMILES = {
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)C(N)c3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "chloramphenicol": "O=[N+]([O-])C(C(O)C(N)CO)c1ccc([N+](=O)[O-])cc1",
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "clonazepam": "NC1=NCCN1C2=NC3=CC(Cl)=C(C=C3C(=O)N2)N(=O)=O",
    "clopidogrel": "COC(=O)[C@H](c1ccccc1Cl)N1CCC(c2sccc2Cl)=C1",
    "digoxin": "CC1OC(OC[C@H]2O[C@H](OC[C@H]3O[C@H](OC4CC[C@]5(C)C(=CC[C@H]4O)CC[C@H]5[C@H]4CC[C@]5(C)C(=O)OC[C@H]45)C[C@H](O)[C@H]3O)[C@H](O)[C@@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@H]1O",
    "doxycycline": "CN(C)c1ccc(C(O)=O)c(O)c1C(=O)c1ccccc1",
    "duloxetine": "CNCCC(Oc1ccc(CS(C)(=O)=O)cc1)c1ccccc1",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "levodopa": "NC(Cc1ccc(O)c(O)c1)C(=O)O",
    "mesalazine": "Nc1cc(O)cc(C(=O)O)c1",
    "metformin": "N=C(N)NC(=N)N",
    "metronidazole": "Cn1c([N+](=O)[O-])cnc1CCO",
    "naproxen": "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",
    "nitrazepam": "O=C1CN=C(c2ccccc2N1)N(=O)=O",
    "omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
    "ranitidine": "CN/C(=N\\NCCSCc1nc[nH]c1C)/NCCN(C)C",
    "sertraline": "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "sorivudine": "O=C1NC(=O)N(C=C1Br)[C@H]1C[C@H](O)[C@@H](CO)O1",
    "sulfasalazine": "O=S(=O)(Nc1ccc(O)cc1)c1ccc(N=Nc2cc(C(=O)O)ccc2O)cc1",
    "warfarin": "CC(C)(O)C(=O)c1c(O)cccc1O",
}


def infer_condition_bucket(value: str) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "mixed"

    rules = [
        ("healthy", "healthy"),
        ("control", "healthy"),
        ("normal", "healthy"),
        ("ibd", "IBD"),
        ("crohn", "IBD"),
        ("ulcerative", "IBD"),
        ("crc", "CRC"),
        ("colorectal", "CRC"),
        ("cancer", "CRC"),
        ("cirrhosis", "cirrhosis"),
        ("cdi", "CDI"),
        ("clostridioides difficile", "CDI"),
        ("t2d", "T2D"),
        ("type 2", "T2D"),
        ("diabetes", "T2D"),
        ("nash", "NASH"),
        ("obes", "obesity"),
    ]
    for needle, bucket in rules:
        if needle in text:
            return bucket
    return "mixed"


def map_microbe_to_taxa_idx(name: str):
    text = str(name).strip().lower()
    for needle, idx in MICROBE_TO_TAXA_IDX.items():
        if needle in text:
            return idx
    return None


def choose_microbiome_row(microbiome, taxa_cols, condition, microbe_idx, microbe_change):
    cond_mask = microbiome["condition"] == condition
    candidates = microbiome[cond_mask]
    if candidates.empty:
        candidates = microbiome

    if microbe_idx is None:
        chosen = candidates.iloc[rng.integers(0, len(candidates))]
        return chosen

    col = taxa_cols[microbe_idx]
    abund = candidates[col].to_numpy(dtype=float)
    if len(abund) == 0:
        chosen = microbiome.iloc[rng.integers(0, len(microbiome))]
        return chosen

    if microbe_change > 0:
        thresh = np.quantile(abund, 0.70)
        focused = candidates[abund >= thresh]
    elif microbe_change < 0:
        thresh = np.quantile(abund, 0.30)
        focused = candidates[abund <= thresh]
    else:
        thresh_lo = np.quantile(abund, 0.35)
        thresh_hi = np.quantile(abund, 0.65)
        focused = candidates[(abund >= thresh_lo) & (abund <= thresh_hi)]

    if focused.empty:
        focused = candidates

    weights = focused[col].to_numpy(dtype=float) + 1e-6
    if microbe_change < 0:
        weights = 1.0 / weights
    weights = weights / weights.sum()
    pick = int(rng.choice(np.arange(len(focused)), p=weights))
    return focused.iloc[pick]


def build_alias_map(substance_info, known_names):
    alias_map = {
        alias.strip().lower(): canonical.strip().lower()
        for alias, canonical in MANUAL_DRUG_ALIASES.items()
    }
    for name in known_names:
        alias_map[name] = name

    if substance_info is None or len(substance_info) == 0:
        return alias_map

    for _, row in substance_info.iterrows():
        terms = []

        primary = str(row.get("Substance_name", "")).strip().lower()
        if primary and primary != "nan":
            terms.append(primary)

        synonyms = str(row.get("synonyms", "")).strip()
        if synonyms and synonyms.lower() != "nan":
            terms.extend(part.strip().lower() for part in synonyms.split(";") if part.strip())

        resolved = None
        for term in terms:
            canonical = alias_map.get(term, term)
            if canonical in known_names:
                resolved = canonical
                break

        if resolved is None:
            continue

        for term in terms:
            alias_map[term] = resolved

    return alias_map


def build_dataset(chembl, microbiome, masi_valid, substance_info, n_pairs):
    print(f"\n[5/6] Building {n_pairs:,} training pairs ...")
    taxa_cols = [f"taxa_{i}" for i in range(N_TAXA)]
    alias_map = build_alias_map(substance_info, set(chembl["_name_clean"]))

    # Build fast MASI drug lookup: drug_name_lower -> list of label dicts
    masi_lookup = {}
    for _, row in masi_valid.iterrows():
        raw_name = str(row["_drug_name_clean"]).strip().lower()
        name = alias_map.get(raw_name, raw_name)
        if name not in masi_lookup:
            masi_lookup[name] = []
        masi_lookup[name].append({
            "effect":         int(row["_effect"]),
            "bioavail_delta": float(row["_bioavail_delta"]),
            "toxicity":       float(row["_toxicity"]),
            "microbe_idx":    map_microbe_to_taxa_idx(row.get("_microbe_clean", "")),
            "microbe_change": float(row.get("_microbe_change", 0.0)),
            "condition_hint": infer_condition_bucket(row.get("Model_Condition/Disease", "")),
        })

    # Split drug pool: MASI-matched vs unmatched
    chembl["_in_masi"] = chembl["_name_clean"].isin(masi_lookup)
    masi_drugs    = chembl[chembl["_in_masi"]].reset_index(drop=True)
    general_drugs = chembl[~chembl["_in_masi"]].reset_index(drop=True)

    print(f"  MASI-matched drugs: {len(masi_drugs):,}")
    print(f"  General drugs:      {len(general_drugs):,}")

    # Push a larger real-label share when the named catalog can support it.
    n_masi_cap = sum(len(masi_lookup[name]) for name in masi_drugs["_name_clean"].unique()) * 30
    n_masi_target = int(n_pairs * 0.55)
    n_masi = min(n_masi_target, n_masi_cap) if len(masi_drugs) else 0
    n_general = n_pairs - n_masi

    rows = []

    # ── MASI-matched pairs (real labels) ─────────────────────────
    for i in range(n_masi):
        d_row  = masi_drugs.iloc[rng.integers(0, len(masi_drugs))]
        name   = str(d_row["_name_clean"])
        labels = masi_lookup[name][rng.integers(0, len(masi_lookup[name]))]
        target_condition = labels["condition_hint"]
        m_row = choose_microbiome_row(
            microbiome=microbiome,
            taxa_cols=taxa_cols,
            condition=target_condition,
            microbe_idx=labels["microbe_idx"],
            microbe_change=labels["microbe_change"],
        )

        condition = str(m_row.get("condition", "healthy"))
        cond_mod  = CONDITION_MOD.get(condition, 0.0)
        base_bio  = BIOAVAIL.get(name, 0.62)

        # Real MASI delta applied on top of base bioavailability
        bio = float(np.clip(
            base_bio + labels["bioavail_delta"] + cond_mod +
            float(rng.normal(0, 0.03)), 0.05, 0.99
        ))

        # Preserve toxic effects as low-response plus elevated toxicity.
        effect = int(labels["effect"])
        resp = 0 if effect < 0 else effect

        tox = float(np.clip(
            labels["toxicity"] + abs(cond_mod) * 0.3 +
            (0.12 if effect < 0 else 0.0) +
            float(rng.normal(0, 0.02)), 0.0, 1.0
        ))

        row = {
            "drug_name":       name,
            "smiles":          str(d_row["smiles"]),
            "microbiome_sample_id": str(m_row["microbiome_sample_id"]),
            "condition":       condition,
            "bioavailability": bio,
            "response_class":  resp,
            "toxicity":        tox,
            "label_source":    "masi_real",
            "masi_effect_raw": effect,
        }
        for j in range(N_TAXA):
            row[f"taxa_{j}"] = float(m_row[f"taxa_{j}"])
        rows.append(row)

    # ── General pairs (biology-formula labels) ───────────────────
    for i in range(n_general):
        d_row  = general_drugs.iloc[rng.integers(0, len(general_drugs))]
        m_row  = microbiome.iloc[rng.integers(0, len(microbiome))]
        name   = str(d_row["_name_clean"])
        condition = str(m_row.get("condition", "healthy"))

        taxa_vals = m_row[taxa_cols].values.astype(float)
        akk   = float(taxa_vals[0])
        bifid = float(taxa_vals[1])
        lacto = float(taxa_vals[2])
        cdiff = float(taxa_vals[3])
        fprau = float(taxa_vals[5]) if len(taxa_vals) > 5 else 0.0
        protective = akk + bifid + lacto + fprau

        base     = BIOAVAIL.get(name, 0.62)
        cond_mod = CONDITION_MOD.get(condition, 0.0)
        drug_mod = 0.06 * protective - 0.04 * cdiff
        bio = float(np.clip(
            base + cond_mod + drug_mod + float(rng.normal(0, 0.04)),
            0.05, 0.99
        ))

        logit = protective * 2.5 - cdiff * 2.0 + (bio - 0.5) * 2.0
        logit += float(rng.normal(0, 0.35))

        shannon = float(-np.sum(taxa_vals * np.log(taxa_vals + 1e-12)))
        tox = float(np.clip(
            0.08 + cdiff * 1.5 + max(0.0, 2.0 - shannon) * 0.03 +
            float(rng.normal(0, 0.02)), 0.0, 1.0
        ))

        row = {
            "drug_name":       name,
            "smiles":          str(d_row["smiles"]),
            "microbiome_sample_id": str(m_row["microbiome_sample_id"]),
            "condition":       condition,
            "bioavailability": bio,
            "response_class":  1,   # will be fixed by percentile below
            "toxicity":        tox,
            "label_source":    "formula",
            "_logit":          logit,
            "masi_effect_raw": 999,
        }
        for j in range(N_TAXA):
            row[f"taxa_{j}"] = float(m_row[f"taxa_{j}"])
        rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  {len(rows):,} / {n_pairs:,} pairs built ...")

    df = pd.DataFrame(rows)

    # Fix response_class for formula rows using percentile thresholds
    formula_mask = df["label_source"] == "formula"
    if formula_mask.sum() > 0:
        logits = df.loc[formula_mask, "_logit"].values
        t1     = np.percentile(logits, 33)
        t2     = np.percentile(logits, 67)
        df.loc[formula_mask, "response_class"] = np.where(
            logits < t1, 0, np.where(logits < t2, 1, 2)
        ).astype(int)

    if "_logit" in df.columns:
        df = df.drop(columns=["_logit"])

    return df


# ─────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────

def save(df):
    print("\n[6/6] Saving ...")
    out = PROCESSED / "training_dataset.csv"
    df.to_csv(out, index=False)

    dist  = df["response_class"].value_counts().sort_index().to_dict()
    cond  = df["condition"].value_counts().to_dict()
    src   = df["label_source"].value_counts().to_dict()
    size  = out.stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"  Training dataset ready")
    print(f"{'='*60}")
    print(f"  Total rows:        {len(df):,}")
    print(f"  Label sources:     {src}")
    print(f"  Response dist:     {dist}")
    print(f"  Conditions:        {cond}")
    print(f"  File size:         {size:.1f} MB")
    print(f"  Saved:             {out}")
    print(f"\n  NOW RUN:")
    print(f"    python train.py --n_epochs 200 --batch_size 32")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MicroDrugNet — Build Real Training Dataset")
    print("===========================================")

    masi_all, masi_valid = load_masi_real()
    substance_info       = load_masi_substance_info()
    chembl               = load_chembl()
    microbiome           = load_microbiome()

    df = build_dataset(chembl, microbiome, masi_valid, substance_info, N_PAIRS)
    save(df)
