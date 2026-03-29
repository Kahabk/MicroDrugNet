# 🧬 MicroDrugNet

> **Cross-Attention Fusion for Pharmacomicrobiomics Prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-orange)]()
[![ArXiv](https://img.shields.io/badge/arXiv-coming_soon-red)]()

## Why MicroDrugNet?

The same drug works for Person A, fails for Person B — because their **gut microbiome metabolises it differently**. This is why 30–40% of clinical trials fail for unexplained reasons.

```
Standard assumption:  Drug → Blood → Target         ✅ well-modelled
Reality:              Drug → Gut Microbiome → Modified Drug → Blood → Target
                               ↑
                       MicroDrugNet fills THIS gap
```

**Pharmacomicrobiomics** is an emerging field with published papers but **zero open-source AI models**. MicroDrugNet is the first.

## Architecture

```
Drug SMILES ──────► DrugGNN (GAT) ──────────────┐
                                                  ▼
                                       CrossAttentionFusion ──► Multi-task heads
                                                  ▲              ├─ Bioavailability (0–1)
Microbiome OTU ──► MicrobiomeEncoder ────────────┘              ├─ Metabolite fingerprint
                                                                 └─ Response class (3)
```

The **Cross-Attention Fusion** module is the core novel contribution: drug molecular graph and microbiome OTU vector mutually attend to each other, learning interaction-specific representations.

## Quick Start

```bash
# Install
git clone https://github.com/YOUR_USERNAME/MicroDrugNet
cd MicroDrugNet
pip install -e .

# Download data
# Note: HMDB may require a short form submission on https://www.hmdb.ca/downloads
# before you can place `hmdb_metabolites.zip` into `data/raw/`.
python data/download_data.py

# Quick test with synthetic data
python train/train.py --synthetic --epochs 5

# Full training
python train/train.py --data data/processed/dataset.pkl --epochs 100 --wandb

# Benchmark vs baselines
python evaluate/benchmark.py --data data/processed/dataset.pkl --ckpt checkpoints/best_model.pt

# Run demo
python demo/app.py --ckpt checkpoints/best_model.pt
```

## Datasets Used

| Source | Content | Access |
|--------|---------|--------|
| DrugBank | 11,000+ drugs with SMILES | Free (academic) |
| HMP | Human Microbiome Project 16S data | Free |
| GMrepo | Gut microbiome + disease links | Free |
| miMDB | Microbiome–drug interaction pairs | Free |
| HMDB | Human Metabolome Database | Free, but download page may require form submission |

## Results (Target)

| Model | AUROC | Bio MAE |
|-------|-------|---------|
| Random Forest | 0.71 | — |
| Simple MLP | 0.74 | — |
| **MicroDrugNet (ours)** | **>0.82** | **<0.12** |

> AUROC > 0.82 = publishable threshold

## Project Structure

```
MicroDrugNet/
├── microdrug/          # Core library
│   ├── model.py        # Full MicroDrugNet architecture
│   ├── dataset.py      # Dataset + DataLoader
│   └── losses.py       # Multi-task loss
├── data/
│   ├── download_data.py
│   └── preprocess.py
├── train/train.py      # Training script (AMP, early stop, W&B)
├── evaluate/benchmark.py
├── demo/app.py         # Gradio demo (HF Spaces ready)
└── configs/default.yaml
```

## Citation

```bibtex
@article{microdrugnet2025,
  title   = {MicroDrugNet: Cross-Attention Fusion for Pharmacomicrobiomics Prediction},
  author  = {Your Name},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## License

MIT — free for academic and commercial use.
# test change
# final change
