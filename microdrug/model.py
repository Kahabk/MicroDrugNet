
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


class DrugGNN(nn.Module):
    def __init__(self, node_features=9, hidden=192, out=384, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False, dropout=dropout)
        self.conv3 = GATConv(hidden, out, heads=4, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.dropout(torch.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(torch.relu(self.bn2(self.conv2(x, edge_index))))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)


class MicrobiomeEncoder(nn.Module):
    def __init__(self, n_taxa=1000, hidden=768, out=384, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_taxa, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out), nn.LayerNorm(out),
        )

    def forward(self, x):
        return self.encoder(torch.log1p(x))


class PatientEncoder(nn.Module):
    def __init__(self, n_genes=512, out=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, out), nn.LayerNorm(out),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention — the novel contribution of MicroDrugNet."""
    def __init__(self, dim=384, heads=8, dropout=0.1):
        super().__init__()
        self.drug_to_micro = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.micro_to_drug = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_d = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, drug_emb, micro_emb):
        d = drug_emb.unsqueeze(1)
        m = micro_emb.unsqueeze(1)
        d_att, _ = self.drug_to_micro(d, m, m)
        m_att, _ = self.micro_to_drug(m, d, d)
        drug_out  = self.norm_d(drug_emb  + self.dropout(d_att.squeeze(1)))
        micro_out = self.norm_m(micro_emb + self.dropout(m_att.squeeze(1)))
        return self.norm_out(self.ffn(torch.cat([drug_out, micro_out], dim=-1)))


class MicroDrugNet(nn.Module):
    def __init__(
        self,
        n_taxa=1000,
        n_genes=512,
        drug_dim=384,
        use_patient=False,
        dropout=0.2,
        n_conditions=9,
        n_drugs=50000,
    ):
        super().__init__()
        self.use_patient   = use_patient
        self.n_taxa        = n_taxa
        self.drug_encoder  = DrugGNN(out=drug_dim, dropout=dropout)
        self.micro_encoder = MicrobiomeEncoder(n_taxa=n_taxa, out=drug_dim, dropout=dropout)
        self.fusion        = CrossAttentionFusion(dim=drug_dim, dropout=dropout)
        self.drug_id_embedding = nn.Embedding(n_drugs + 1, drug_dim, padding_idx=0)
        self.condition_proj = nn.Sequential(
            nn.Linear(n_conditions, drug_dim), nn.LayerNorm(drug_dim), nn.GELU()
        )
        # The current training CSV is driven by a small set of taxa and
        # condition-dependent rules, so expose those summary signals directly.
        self.micro_summary_proj = nn.Sequential(
            nn.Linear(10, drug_dim), nn.LayerNorm(drug_dim), nn.GELU(), nn.Dropout(dropout)
        )
        if use_patient:
            self.patient_encoder = PatientEncoder(n_genes=n_genes, out=drug_dim)
            self.patient_gate    = nn.Sequential(
                nn.Linear(drug_dim * 2, drug_dim), nn.Sigmoid()
            )
        # All heads output RAW LOGITS — sigmoid/softmax applied at inference only.
        # This is required for AMP (autocast) compatibility.
        self.head_bioavail = nn.Sequential(nn.Linear(drug_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_response = nn.Sequential(nn.Linear(drug_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.head_toxicity = nn.Sequential(nn.Linear(drug_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_metabol  = nn.Sequential(nn.Linear(drug_dim, 256), nn.ReLU(), nn.Linear(256, 512))

    def forward(self, drug_graph, microbiome_profile, patient_data=None, condition_feat=None, drug_idx=None):
        drug_emb  = self.drug_encoder(drug_graph.x, drug_graph.edge_index, drug_graph.batch)
        micro_emb = self.micro_encoder(microbiome_profile)
        fused     = self.fusion(drug_emb, micro_emb)
        if drug_idx is not None:
            fused = fused + 0.25 * self.drug_id_embedding(drug_idx)
        summary_input = self._build_micro_summary(microbiome_profile)
        fused = fused + 0.30 * self.micro_summary_proj(summary_input)
        if condition_feat is not None:
            fused = fused + 0.35 * self.condition_proj(condition_feat)
        if self.use_patient and patient_data is not None:
            pat_emb = self.patient_encoder(patient_data)
            gate    = self.patient_gate(torch.cat([fused, pat_emb], dim=-1))
            fused   = fused * gate + pat_emb * (1 - gate)
        bio_logit = self.head_bioavail(fused).squeeze(-1)
        tox_logit = self.head_toxicity(fused).squeeze(-1)
        return {
            # Raw logits for loss computation
            "bioavailability_logit": bio_logit,
            "response_class":        self.head_response(fused),
            "toxicity_logit":         tox_logit,
            "metabolites":            self.head_metabol(fused),
            # Sigmoid outputs for inference/evaluation
            "bioavailability":        torch.sigmoid(bio_logit),
            "toxicity":               torch.sigmoid(tox_logit),
        }

    def _build_micro_summary(self, microbiome_profile):
        x = microbiome_profile
        signal = x[:, :7]
        diversity = -(x * torch.log(x + 1e-12)).sum(dim=-1, keepdim=True)
        richness = (x > 0).float().mean(dim=-1, keepdim=True)
        total = x.sum(dim=-1, keepdim=True)
        return torch.cat([signal, diversity, richness, total], dim=-1)
