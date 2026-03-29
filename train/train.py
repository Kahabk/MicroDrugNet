"""
MicroDrugNet — Production Training Script
Usage:
  python train/train.py --synthetic --epochs 5   # quick test
  python train/train.py --data data/processed/dataset.pkl --epochs 100 --wandb
"""
import os, sys, time, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.nn as nn
from torch.cuda.amp import GradScaler
from microdrug.model   import MicroDrugNet
from microdrug.losses  import MicroDrugLoss
from microdrug.dataset import get_loaders, load_dataset

try:
    import wandb; HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_response_labels(batch, device):
    key = "response" if "response" in batch else "response_class"
    labels = batch[key].to(device)
    return {"response": labels, "response_class": labels}


def get_response_logits(preds):
    if "response_logits" in preds:
        return preds["response_logits"]
    if "response_class" in preds:
        return preds["response_class"]
    raise KeyError("Model output is missing response logits")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          type=str,   default=None)
    p.add_argument("--synthetic",     action="store_true")
    p.add_argument("--n-taxa",        type=int,   default=1000)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-5)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--ckpt-dir",      type=str,   default="checkpoints")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--workers",       type=int,   default=0)
    return p.parse_args()

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train(); total, n = 0.0, 0
    for batch in loader:
        drug   = batch["drug"].to(device)
        micro  = batch["microbiome"].to(device)
        labels = {"bioavailability": batch["bioavailability"].to(device), **get_response_labels(batch, device)}
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            preds = model(drug, micro)
            loss, _ = criterion(preds, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item(); n += 1
    return total / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); total, n = 0.0, 0
    bp, bgt, rp, rgt = [], [], [], []
    for batch in loader:
        drug   = batch["drug"].to(device)
        micro  = batch["microbiome"].to(device)
        labels = {"bioavailability": batch["bioavailability"].to(device), **get_response_labels(batch, device)}
        preds  = model(drug, micro)
        loss, _ = criterion(preds, labels)
        total += loss.item(); n += 1
        bp.extend(preds["bioavailability"].cpu().tolist())
        bgt.extend(labels["bioavailability"].cpu().tolist())
        rp.extend(get_response_logits(preds).argmax(-1).cpu().tolist())
        rgt.extend(labels["response"].cpu().tolist())
    from sklearn.metrics import mean_absolute_error, accuracy_score
    return total/n, mean_absolute_error(bgt, bp), accuracy_score(rgt, rp)

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.synthetic:
        from data.preprocess import make_synthetic
        records = make_synthetic(n_taxa=args.n_taxa)
    elif args.data:
        records = load_dataset(args.data)
    else:
        raise ValueError("--synthetic or --data required")
    trn, val, tst = get_loaders(records, batch_size=args.batch, num_workers=args.workers)
    model     = MicroDrugNet(n_taxa=args.n_taxa).to(device)
    criterion = MicroDrugLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler() if args.amp and device.type == "cuda" else None
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    if args.wandb and HAS_WANDB:
        wandb.init(project="MicroDrugNet", config=vars(args))
    ckpt = Path(args.ckpt_dir); ckpt.mkdir(exist_ok=True)
    best_val, pat = float("inf"), 0
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tl = train_epoch(model, trn, optimizer, criterion, scaler, device)
        vl, vm, va = eval_epoch(model, val, criterion, device)
        scheduler.step()
        print(f"Ep {ep:03d} | trn={tl:.4f} val={vl:.4f} bio_mae={vm:.4f} resp_acc={va:.4f} | {time.time()-t0:.1f}s")
        if args.wandb and HAS_WANDB:
            wandb.log({"train_loss":tl,"val_loss":vl,"bio_mae":vm,"resp_acc":va,"epoch":ep})
        if vl < best_val:
            best_val = vl; pat = 0
            torch.save(model.state_dict(), ckpt/"best_model.pt")
        else:
            pat += 1
            if pat >= args.patience:
                print(f"Early stop @ epoch {ep}"); break
    model.load_state_dict(torch.load(ckpt/"best_model.pt", weights_only=True))
    tl2, tm, ta = eval_epoch(model, tst, criterion, device)
    print(f"\nTEST → loss={tl2:.4f} bio_mae={tm:.4f} resp_acc={ta:.4f}")
    json.dump({"test_loss":tl2,"bio_mae":tm,"resp_acc":ta}, open(ckpt/"results.json","w"), indent=2)

if __name__ == "__main__":
    main()
