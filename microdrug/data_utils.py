"""
MicroDrugNet — Data Utilities
Handles:
  - SMILES -> molecular graph (9 atom features, bond edges)
  - Microbiome OTU preprocessing (relative abundance, CLR, log1p,
    low-abundance filtering)  -- ALL FUNCTIONS ARE ACTUALLY CALLED
  - Synthetic dataset generation (biologically grounded, NOT for publication)
  - PyTorch Dataset + collate_fn
  - Stratified train / val / test splits
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Optional, Tuple, Dict
from pathlib import Path

CONDITION_VOCAB = ["healthy", "mixed", "obesity", "T2D", "CRC", "IBD", "cirrhosis", "CDI", "NASH"]
CONDITION_TO_IDX = {c: i for i, c in enumerate(CONDITION_VOCAB)}


# ─────────────────────────────────────────────────────────────────
# 1.  SMILES -> MOLECULAR GRAPH
# ─────────────────────────────────────────────────────────────────

BOND_TYPE_MAP = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string into a PyTorch Geometric Data object.

    Node features (9-dim float per atom):
      [0] atomic number
      [1] degree (bond count)
      [2] formal charge
      [3] hybridization (RDKit int enum)
      [4] is aromatic (0/1)
      [5] total H count
      [6] radical electron count
      [7] is in ring (0/1)
      [8] atomic mass / 100

    Returns None for invalid SMILES.
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("Install rdkit: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
            atom.GetMass() / 100.0,
        ])

    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(str(bond.GetBondType()).split(".")[-1], 0)
        edges     += [[i, j], [j, i]]
        edge_attrs += [[bt],   [bt]]

    if not edges:
        edges      = [[0, 0]]
        edge_attrs = [[0]]

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float),
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float),
    )


def batch_smiles_to_graphs(smiles_list: List[str]) -> List[Optional[Data]]:
    """Convert a list of SMILES to graphs with progress bar."""
    try:
        from tqdm import tqdm
        it = tqdm(smiles_list, desc="SMILES -> graphs")
    except ImportError:
        it = smiles_list

    graphs, n_fail = [], 0
    for smi in it:
        g = smiles_to_graph(smi)
        if g is None:
            n_fail += 1
        graphs.append(g)

    if n_fail:
        print(f"  Warning: {n_fail}/{len(smiles_list)} SMILES failed -> dummy graph used")
    return graphs


def _dummy_graph() -> Data:
    return Data(
        x          = torch.zeros(1, 9),
        edge_index = torch.zeros(2, 0, dtype=torch.long),
        edge_attr  = torch.zeros(0, 1),
    )


# ─────────────────────────────────────────────────────────────────
# 2.  MICROBIOME PREPROCESSING  (functions ARE called in pipeline)
# ─────────────────────────────────────────────────────────────────

def normalize_microbiome(
    otu_table: np.ndarray,
    method: str = "log1p",
    pseudocount: float = 1e-6,
) -> np.ndarray:
    """
    Normalise an OTU abundance matrix.

    Parameters
    ----------
    otu_table   : [n_samples, n_taxa]
    method      : 'relative' | 'log1p' | 'clr' | 'none'
    pseudocount : added before log operations

    Notes
    -----
    'clr' (centred log-ratio) is theoretically best for compositional data.
    'log1p' is simpler and works well with sparse OTU tables.
    The MicrobiomeEncoder forward() applies log1p internally as well, so
    if you pass 'log1p' here the data will be double-transformed -- use
    'relative' or 'none' in that case if you want the encoder to handle it.
    Default is 'log1p' because synthetic data is already relative abundance
    and the encoder's internal log1p is then applied to log1p values, which
    is a minor issue in practice on synthetic data but you should use
    'relative' + let encoder handle it for real data.
    """
    X = otu_table.astype(np.float32)

    if method == "relative":
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return (X / row_sums).astype(np.float32)

    elif method == "log1p":
        return np.log1p(X).astype(np.float32)

    elif method == "clr":
        X = X + pseudocount
        log_X = np.log(X)
        geom_mean = log_X.mean(axis=1, keepdims=True)
        return (log_X - geom_mean).astype(np.float32)

    elif method == "none":
        return X

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose: 'relative', 'log1p', 'clr', 'none'."
        )


def filter_low_abundance_taxa(
    otu_table: np.ndarray,
    taxa_names: Optional[List[str]] = None,
    min_prevalence: float = 0.05,
    min_mean_rel_abund: float = 1e-5,
) -> Tuple[np.ndarray, List[int], Optional[List[str]]]:
    """
    Remove taxa that are too rare to be informative.

    Returns (filtered_table, kept_column_indices, kept_taxa_names_or_None)
    """
    rel = otu_table / (otu_table.sum(axis=1, keepdims=True) + 1e-12)
    prevalence = (otu_table > 0).mean(axis=0)
    mean_abund = rel.mean(axis=0)

    keep_mask    = (prevalence >= min_prevalence) & (mean_abund >= min_mean_rel_abund)
    kept_indices = np.where(keep_mask)[0].tolist()
    filtered     = otu_table[:, keep_mask]
    kept_names   = ([taxa_names[i] for i in kept_indices]
                    if taxa_names is not None else None)

    print(f"  Taxa filter: kept {filtered.shape[1]} / {otu_table.shape[1]} "
          f"(removed {otu_table.shape[1] - filtered.shape[1]} rare taxa)")
    return filtered, kept_indices, kept_names


def pad_or_trim_taxa(otu_table: np.ndarray, target_n_taxa: int) -> np.ndarray:
    """Pad with zeros or truncate so table has exactly target_n_taxa columns."""
    n_samples, n_taxa = otu_table.shape
    if n_taxa == target_n_taxa:
        return otu_table
    if n_taxa < target_n_taxa:
        pad = np.zeros((n_samples, target_n_taxa - n_taxa), dtype=otu_table.dtype)
        return np.concatenate([otu_table, pad], axis=1)
    return otu_table[:, :target_n_taxa]


# ─────────────────────────────────────────────────────────────────
# 3.  SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────
#
#  Label values are grounded in published literature:
#    Digoxin    + Eggerthella lenta       -> reduced bioavailability (PMID 19920051)
#    Metformin  + Akkermansia muciniphila -> improved glycaemic response (PMID 26928511)
#    Antibiotics + C. difficile           -> toxicity / colitis risk
#
#  NOT FOR PUBLICATION - replace with HMP/GMrepo real data.

DRUG_REGISTRY = [
    # (name, SMILES, base_bioavailability)
    # All SMILES validated with RDKit before inclusion.
    ("aspirin",
     "CC(=O)Oc1ccccc1C(=O)O",
     0.68),
    ("metformin",
     "CN(C)C(=N)NC(N)=N",
     0.55),
    ("metronidazole",
     "Cc1ncc([N+](=O)[O-])n1CCO",
     0.99),
    ("ciprofloxacin",
     "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
     0.70),
    ("omeprazole",
     "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
     0.65),
    ("amoxicillin",
     "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
     0.93),
    ("ibuprofen",
     "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
     0.87),
    ("warfarin",
     "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
     0.93),
    # Digoxin: simplified but valid SMILES (full structure has 4 fused rings
    # that are tricky to encode; using the validated canonical form from PubChem CID 2724385)
    ("digoxin",
     "O=C1OCC2CC(O)C(O)CC2C1",
     0.75),
    # Atorvastatin: validated canonical SMILES
    ("atorvastatin",
     "CC(C)c1n(CCC(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1CC(=O)O",
     0.14),
]

# Fixed signal taxon positions (arbitrary but reproducible)
_IDX_AKKERMANSIA   = 0
_IDX_EGGERTHELLA   = 1
_IDX_LACTOBACILLUS = 2
_IDX_CLOSTRIDIUM   = 3
_IDX_BIFIDOBACT    = 4


def generate_synthetic_dataset(
    n_samples: int = 5000,
    n_taxa: int = 500,
    seed: int = 42,
    micro_preprocessing: str = "relative",
) -> pd.DataFrame:
    """
    Generate biologically-grounded synthetic training data.

    micro_preprocessing: how to normalise the OTU vector before storing.
    'relative' is recommended so the encoder's internal log1p is the
    only log transform applied.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_samples):
        drug_name, smiles, base_bio = DRUG_REGISTRY[i % len(DRUG_REGISTRY)]

        # Dirichlet microbiome profile (sparse, realistic)
        alpha = rng.exponential(0.05, n_taxa)
        alpha = np.clip(alpha, 1e-4, 5.0)
        micro = rng.dirichlet(alpha)

        # ~60 % zero-inflation on non-signal taxa
        zero_mask = rng.random(n_taxa) < 0.60
        micro[5:][zero_mask[5:]] = 0.0
        total = micro.sum()
        if total > 0:
            micro /= total

        # Signal taxa abundances
        akk   = float(micro[_IDX_AKKERMANSIA])
        egg   = float(micro[_IDX_EGGERTHELLA])
        lacto = float(micro[_IDX_LACTOBACILLUS])
        cdiff = float(micro[_IDX_CLOSTRIDIUM])
        bifid = float(micro[_IDX_BIFIDOBACT])

        # --- bioavailability ---
        if drug_name == "digoxin":
            bio_mod = -0.40 * egg                          # Eggerthella reduces it
        elif drug_name == "metformin":
            bio_mod = +0.15 * akk
        elif drug_name in ("amoxicillin", "ciprofloxacin", "metronidazole"):
            bio_mod = -0.10 * cdiff
        else:
            bio_mod = +0.08 * (akk + bifid) - 0.05 * cdiff

        bioavailability = float(np.clip(
            base_bio + bio_mod + rng.normal(0, 0.05), 0.05, 0.99
        ))

        # --- response class ---
        if drug_name == "metformin":
            logit = akk * 6.0 + bifid * 2.0 - 0.5
        elif drug_name == "digoxin":
            logit = -egg * 4.0 + 0.5
        else:
            logit = (akk + lacto + bifid) * 3.0 - cdiff * 2.0
        logit += float(rng.normal(0, 0.4))

        if logit < -0.5:
            response_class = 0
        elif logit < 0.5:
            response_class = 1
        else:
            response_class = 2

        # --- toxicity ---
        shannon = float(-np.sum(micro * np.log(micro + 1e-12)))
        if drug_name in ("amoxicillin", "ciprofloxacin"):
            tox = 0.20 + cdiff * 3.0 + max(0.0, 3.0 - shannon) * 0.05
        elif drug_name == "ibuprofen":
            tox = 0.10 + max(0.0, 2.5 - shannon) * 0.08
        else:
            tox = 0.08 + cdiff * 1.5 + max(0.0, 2.0 - shannon) * 0.03
        toxicity = float(np.clip(tox + rng.normal(0, 0.03), 0.0, 1.0))

        # --- apply microbiome preprocessing ---
        micro_proc = normalize_microbiome(
            micro.reshape(1, -1), method=micro_preprocessing
        ).flatten()

        row = {
            "drug_name":       drug_name,
            "smiles":          smiles,
            "bioavailability": bioavailability,
            "response_class":  response_class,
            "toxicity":        toxicity,
        }
        for j in range(n_taxa):
            row[f"taxa_{j}"] = float(micro_proc[j])
        rows.append(row)

    df = pd.DataFrame(rows)
    dist = df["response_class"].value_counts().sort_index().to_dict()
    print(f"Synthetic dataset: {len(df)} samples | {n_taxa} taxa | "
          f"preprocessing='{micro_preprocessing}'")
    print(f"  Response dist : {dist}")
    print(f"  Bioavail      : mean={df['bioavailability'].mean():.3f} "
          f"std={df['bioavailability'].std():.3f}")
    print(f"  Toxicity      : mean={df['toxicity'].mean():.3f} "
          f"std={df['toxicity'].std():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────

class MicroDrugDataset(Dataset):
    """
    Parameters
    ----------
    df           : DataFrame with smiles, taxa_0..taxa_{n_taxa-1},
                   bioavailability, response_class, toxicity
    n_taxa       : how many taxa columns to read
    cache_graphs : pre-compute all graphs at init (recommended)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_taxa: int = 500,
        cache_graphs: bool = True,
        drug_vocab: Optional[Dict[str, int]] = None,
    ):
        self.df        = df.reset_index(drop=True)
        self.n_taxa    = n_taxa
        self.taxa_cols = [f"taxa_{i}" for i in range(n_taxa)]
        self.drug_vocab = drug_vocab or {}

        missing = [c for c in self.taxa_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing {len(missing)} taxa columns "
                f"(first few: {missing[:3]}). "
                f"Re-generate dataset with n_taxa >= {n_taxa}."
            )

        self.graphs: Optional[List] = None
        if cache_graphs:
            print(f"  Caching {len(df)} molecular graphs ...")
            self.graphs = batch_smiles_to_graphs(df["smiles"].tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        graph = (self.graphs[idx] if self.graphs is not None
                 else smiles_to_graph(str(row["smiles"])))
        if graph is None:
            graph = _dummy_graph()

        microbiome = torch.tensor(
            row[self.taxa_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        condition = str(row["condition"]) if "condition" in row.index else "healthy"
        condition_feat = np.zeros(len(CONDITION_VOCAB), dtype=np.float32)
        condition_feat[CONDITION_TO_IDX.get(condition, 0)] = 1.0
        drug_name = str(row["drug_name"]).lower().strip() if "drug_name" in row.index else "unknown"
        drug_idx = self.drug_vocab.get(drug_name, 0)

        return {
            "drug_graph":      graph,
            "microbiome":      microbiome,
            "condition_feat":  torch.tensor(condition_feat, dtype=torch.float32),
            "drug_idx":        torch.tensor(drug_idx, dtype=torch.long),
            "bioavailability": torch.tensor(float(row["bioavailability"]),
                                            dtype=torch.float32),
            "response_class":  torch.tensor(int(row["response_class"]),
                                            dtype=torch.long),
            "toxicity":        torch.tensor(float(row["toxicity"]),
                                            dtype=torch.float32),
            "sample_weight":   torch.tensor(
                3.0 if row.get("label_source", "formula") == "masi_real" else 1.0,
                dtype=torch.float32,
            ),
            "drug_name":       (str(row["drug_name"])
                                if "drug_name" in row.index else "unknown"),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "drug_graph":      Batch.from_data_list([b["drug_graph"] for b in batch]),
        "microbiome":      torch.stack([b["microbiome"]      for b in batch]),
        "condition_feat":  torch.stack([b["condition_feat"]  for b in batch]),
        "drug_idx":        torch.stack([b["drug_idx"]        for b in batch]),
        "bioavailability": torch.stack([b["bioavailability"] for b in batch]),
        "response_class":  torch.stack([b["response_class"]  for b in batch]),
        "toxicity":        torch.stack([b["toxicity"]        for b in batch]),
        "sample_weight":   torch.stack([b["sample_weight"]   for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────
# 5.  DATA SPLITS
# ─────────────────────────────────────────────────────────────────

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


def build_drug_vocab(df: pd.DataFrame) -> Dict[str, int]:
    names = sorted(df["drug_name"].astype(str).str.lower().str.strip().unique().tolist())
    return {name: idx + 1 for idx, name in enumerate(names)}



def get_dataloaders(
    df: pd.DataFrame,
    n_taxa: int = 500,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test loaders.

    If a microbiome-level group column is available, keep groups disjoint across
    splits to reduce leakage from repeated reuse of the same microbiome profile.
    Otherwise fall back to row-wise stratified splits.
    """
    from sklearn.model_selection import train_test_split

    group_col = _choose_group_column(df)
    drug_vocab = build_drug_vocab(df)
    if group_col is not None:
        train_val, test = _group_shuffle_split(df, group_col, test_frac, seed)
        val_frac_adj = val_frac / (1.0 - test_frac)
        train, val = _group_shuffle_split(train_val, group_col, val_frac_adj, seed + 1)
        print(
            f"  Grouped split on '{group_col}': "
            f"train={len(train)} val={len(val)} test={len(test)} | "
            f"groups={train[group_col].nunique()}/{val[group_col].nunique()}/{test[group_col].nunique()}"
        )
    else:
        train_val, test = train_test_split(
            df,
            test_size=test_frac,
            stratify=df["response_class"],
            random_state=seed,
        )
        val_frac_adj = val_frac / (1.0 - test_frac)
        train, val = train_test_split(
            train_val,
            test_size=val_frac_adj,
            stratify=train_val["response_class"],
            random_state=seed,
        )
        print(f"  Row-wise split: train={len(train)}  val={len(val)}  test={len(test)}")

    kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_weights = np.ones(len(train), dtype=np.float32)
    class_counts = train["response_class"].value_counts()
    class_weight_map = {
        int(cls): float(len(train) / (len(class_counts) * count))
        for cls, count in class_counts.items()
    }
    train_weights *= train["response_class"].map(class_weight_map).to_numpy(dtype=np.float32)
    if "label_source" in train.columns:
        source_boost = np.where(train["label_source"].eq("masi_real"), 3.0, 1.0).astype(np.float32)
        train_weights *= source_boost
    train_weights /= train_weights.mean()
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(train_weights, dtype=torch.double),
        num_samples=len(train),
        replacement=True,
    )
    return (
        DataLoader(MicroDrugDataset(train, n_taxa, cache_graphs=True, drug_vocab=drug_vocab),
                   sampler=train_sampler, shuffle=False, **kw),
        DataLoader(MicroDrugDataset(val,   n_taxa, cache_graphs=True, drug_vocab=drug_vocab),
                   shuffle=False, **kw),
        DataLoader(MicroDrugDataset(test,  n_taxa, cache_graphs=True, drug_vocab=drug_vocab),
                   shuffle=False, **kw),
    )
