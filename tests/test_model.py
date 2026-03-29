"""
MicroDrugNet — Unit Tests
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest, torch
from torch_geometric.data import Batch

def make_dummy_batch(B=4, n_atoms=10, n_taxa=1000):
    from microdrug.model import MicroDrugNet
    from torch_geometric.data import Data
    graphs = []
    for _ in range(B):
        x  = torch.randn(n_atoms, 9)
        ei = torch.randint(0, n_atoms, (2, n_atoms * 2))
        graphs.append(Data(x=x, edge_index=ei))
    batch  = Batch.from_data_list(graphs)
    micro  = torch.rand(B, n_taxa)
    return batch, micro

def test_forward_pass():
    from microdrug.model import MicroDrugNet
    model = MicroDrugNet(n_taxa=1000)
    model.eval()
    drug, micro = make_dummy_batch(B=4)
    with torch.no_grad():
        out = model(drug, micro)
    assert "bioavailability"  in out
    assert "metabolites"      in out
    assert "response_logits"  in out
    assert out["bioavailability"].shape == (4,)
    assert out["metabolites"].shape     == (4, 512)
    assert out["response_logits"].shape == (4, 3)
    assert (out["bioavailability"] >= 0).all() and (out["bioavailability"] <= 1).all()

def test_drug_gnn():
    from microdrug.model import DrugGNN
    from torch_geometric.data import Data, Batch
    model = DrugGNN()
    x  = torch.randn(8, 9)
    ei = torch.randint(0, 8, (2, 16))
    g  = Batch.from_data_list([Data(x=x, edge_index=ei)])
    out = model(g.x, g.edge_index, g.batch)
    assert out.shape == (1, 256)

def test_microbiome_encoder():
    from microdrug.model import MicrobiomeEncoder
    enc = MicrobiomeEncoder(n_taxa=500)
    x   = torch.rand(8, 500)
    out = enc(x)
    assert out.shape == (8, 256)

def test_cross_attention_fusion():
    from microdrug.model import CrossAttentionFusion
    fusion = CrossAttentionFusion(dim=256, heads=8)
    d = torch.randn(4, 256)
    m = torch.randn(4, 256)
    out = fusion(d, m)
    assert out.shape == (4, 256)

def test_smiles_to_graph():
    from data.preprocess import smiles_to_graph
    g = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    assert g is not None
    assert g.x.shape[1] == 9
    g_invalid = smiles_to_graph("NOT_A_SMILES_XYZ")
    assert g_invalid is None

def test_losses():
    from microdrug.losses import MicroDrugLoss
    from microdrug.model  import MicroDrugNet
    model = MicroDrugNet(n_taxa=100)
    drug, micro = make_dummy_batch(B=4, n_taxa=100)
    preds  = model(drug, micro)
    labels = {"bioavailability": torch.rand(4), "response": torch.randint(0,3,(4,))}
    crit   = MicroDrugLoss()
    loss, sub = crit(preds, labels)
    assert loss.item() > 0
    assert "bio" in sub and "resp" in sub
