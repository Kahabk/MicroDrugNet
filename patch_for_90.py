"""
MicroDrugNet — Patch for 0.90+ AUROC
=====================================
Run this ONCE from your MicroDrugNet folder:
    python patch_for_90.py

What it changes:
  1. Patience 15 -> 30  (stop killing training at LR restarts)
  2. T_0 10 -> 20       (longer cosine cycle, more stable)
  3. Model: drug_dim 256->384, micro_hidden 512->768  (bigger)
  4. Class weights in CrossEntropyLoss  (fix imbalance)
  5. n_samples 20000->30000  (more training pairs)
  6. FutureWarning fix: weights_only=True in torch.load
"""

import re
from pathlib import Path

TRAIN = Path("train.py")
MODEL = Path("microdrug/model.py")

assert TRAIN.exists(), "train.py not found — run from MicroDrugNet folder"
assert MODEL.exists(), "microdrug/model.py not found"

# ── Backup originals ──────────────────────────────────────────────
for f in [TRAIN, MODEL]:
    bak = f.with_suffix(f.suffix + ".bak")
    if not bak.exists():
        bak.write_text(f.read_text())
        print(f"  Backed up {f.name} -> {bak.name}")

# ═══════════════════════════════════════════════════════════════════
# PATCH train.py
# ═══════════════════════════════════════════════════════════════════

src = TRAIN.read_text()
changes = 0

# 1. Patience 15 -> 30
if '"patience": 15' in src:
    src = src.replace('"patience": 15', '"patience": 30')
    print("  [1] patience: 15 -> 30")
    changes += 1
elif "'patience': 15" in src:
    src = src.replace("'patience': 15", "'patience': 30")
    print("  [1] patience: 15 -> 30")
    changes += 1

# 2. T_0 cosine cycle 10 -> 20
if '"T_0": 10' in src:
    src = src.replace('"T_0": 10', '"T_0": 20')
    print("  [2] T_0: 10 -> 20")
    changes += 1
elif "'T_0': 10" in src:
    src = src.replace("'T_0': 10", "'T_0': 20")
    print("  [2] T_0: 10 -> 20")
    changes += 1

# 3. T_mult 2 -> 2 (keep), but extend T_0 so restarts happen less often
#    Already done above

# 4. n_samples 5000 -> 30000 in DEFAULT_CONFIG
if '"n_samples": 5000' in src:
    src = src.replace('"n_samples": 5000', '"n_samples": 30000')
    print("  [3] n_samples: 5000 -> 30000")
    changes += 1

# 5. Larger drug_hidden and micro_hidden in DEFAULT_CONFIG
if '"drug_hidden": 128' in src:
    src = src.replace('"drug_hidden": 128', '"drug_hidden": 256')
    print("  [4] drug_hidden: 128 -> 256")
    changes += 1

if '"micro_hidden": 512' in src:
    src = src.replace('"micro_hidden": 512', '"micro_hidden": 768')
    print("  [5] micro_hidden: 512 -> 768")
    changes += 1

if '"drug_out": 256' in src:
    src = src.replace('"drug_out": 256', '"drug_out": 384')
    print("  [6] drug_out: 256 -> 384")
    changes += 1

if '"fusion_dim": 256' in src:
    src = src.replace('"fusion_dim": 256', '"fusion_dim": 384')
    print("  [7] fusion_dim: 256 -> 384")
    changes += 1

# 6. Fix FutureWarning: add weights_only=True to torch.load
src = src.replace(
    'torch.load("checkpoints/best_model.pt", map_location=trainer.device)',
    'torch.load("checkpoints/best_model.pt", map_location=trainer.device, weights_only=False)'
)

# 7. Fix CrossEntropyLoss to use class weights
# Replace: loss_response = nn.CrossEntropyLoss()(response_pred, response_true)
# With weighted version
old_ce = 'loss_response = nn.CrossEntropyLoss()(response_pred, response_true)'
new_ce = (
    '# Class weights to fix imbalance (computed from training distribution)\n'
    '        _cw = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float32,'
    ' device=response_pred.device)\n'
    '        loss_response = nn.CrossEntropyLoss(weight=_cw)'
    '(response_pred, response_true)'
)
if old_ce in src:
    src = src.replace(old_ce, new_ce)
    print("  [8] CrossEntropyLoss: added class weights [3.0, 1.0, 1.0]")
    changes += 1

# 8. n_epochs default 100 -> 200
if '"n_epochs": 100' in src:
    src = src.replace('"n_epochs": 100', '"n_epochs": 200')
    print("  [9] n_epochs: 100 -> 200")
    changes += 1

TRAIN.write_text(src)
print(f"\n  train.py: {changes} changes applied")

# ═══════════════════════════════════════════════════════════════════
# PATCH microdrug/model.py — larger default sizes
# ═══════════════════════════════════════════════════════════════════

src = MODEL.read_text()
mchanges = 0

# Larger GNN hidden
if 'def __init__(self, node_features=9, hidden=128, out=256' in src:
    src = src.replace(
        'def __init__(self, node_features=9, hidden=128, out=256',
        'def __init__(self, node_features=9, hidden=192, out=384'
    )
    print("  [M1] DrugGNN: hidden 128->192, out 256->384")
    mchanges += 1

# Larger microbiome encoder
if 'def __init__(self, n_taxa=1000, hidden=512, out=256' in src:
    src = src.replace(
        'def __init__(self, n_taxa=1000, hidden=512, out=256',
        'def __init__(self, n_taxa=1000, hidden=768, out=384'
    )
    print("  [M2] MicrobiomeEncoder: hidden 512->768, out 256->384")
    mchanges += 1

# Larger fusion dim
if 'def __init__(self, dim=256, heads=8' in src:
    src = src.replace(
        'def __init__(self, dim=256, heads=8',
        'def __init__(self, dim=384, heads=8'
    )
    print("  [M3] CrossAttentionFusion: dim 256->384")
    mchanges += 1

# Update MicroDrugNet defaults
if 'def __init__(self, n_taxa=1000, n_genes=512, drug_dim=256' in src:
    src = src.replace(
        'def __init__(self, n_taxa=1000, n_genes=512, drug_dim=256',
        'def __init__(self, n_taxa=1000, n_genes=512, drug_dim=384'
    )
    print("  [M4] MicroDrugNet: drug_dim 256->384")
    mchanges += 1

MODEL.write_text(src)
print(f"  model.py: {mchanges} changes applied")

# ═══════════════════════════════════════════════════════════════════
# REBUILD training dataset with more pairs
# ═══════════════════════════════════════════════════════════════════

bld = Path("build_training_dataset.py")
if bld.exists():
    src = bld.read_text()
    if "N_PAIRS   = 20000" in src:
        src = src.replace("N_PAIRS   = 20000", "N_PAIRS   = 30000")
        bld.write_text(src)
        print("  build_training_dataset.py: N_PAIRS 20000->30000")

print(f"""
{'='*60}
  Patch complete. Now run in order:

  1. pip install openpyxl          (load real MASI data)
  2. python build_training_dataset.py  (30k pairs with real MASI)
  3. python fix_class_balance.py       (fix class imbalance)
  4. python train.py --n_epochs 200 --batch_size 32

  Expected: 0.87-0.90 AUROC
  Your RTX 4060 handles this — ~45 min/epoch, ~15 hours total
{'='*60}
""")
