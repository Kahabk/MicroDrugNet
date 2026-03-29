"""
MicroDrugNet — PyTorch Dataset + DataLoader utilities
"""
import pickle, torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

class MicroDrugDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    drug_graphs = Batch.from_data_list([b["drug_graph"] for b in batch])
    microbiome  = torch.stack([b["microbiome"] for b in batch])
    bioavail    = torch.tensor([b["bioavailability"] for b in batch], dtype=torch.float)
    response_key = "response" if "response" in batch[0] else "response_class"
    response     = torch.tensor([b[response_key] for b in batch], dtype=torch.long)
    return {
        "drug":          drug_graphs,
        "microbiome":    microbiome,
        "bioavailability": bioavail,
        "response":      response,
        "response_class": response,
    }


def get_loaders(records, val_split=0.15, test_split=0.1, batch_size=32, num_workers=4, seed=42):
    n     = len(records)
    n_val = int(n * val_split)
    n_tst = int(n * test_split)
    n_trn = n - n_val - n_tst

    g = torch.Generator().manual_seed(seed)
    trn, val, tst = torch.utils.data.random_split(
        MicroDrugDataset(records), [n_trn, n_val, n_tst], generator=g
    )
    kw = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(trn, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(tst, batch_size=batch_size, shuffle=False, **kw),
    )


def load_dataset(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
