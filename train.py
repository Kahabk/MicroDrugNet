"""
MicroDrugNet Training Script
Production-ready training with:
- Multi-task loss balancing
- Learning rate scheduling
- Gradient clipping
- Mixed precision (AMP)
- Checkpoint saving
- Early stopping
- W&B logging (optional)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Optional
from torch.amp import GradScaler, autocast


# ─────────────────────────────────────────
# MULTI-TASK LOSS
# ─────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., 2018).
    Learns task weights automatically — no manual tuning needed.

    IMPORTANT: toxicity_pred must be RAW LOGITS (no sigmoid applied).
    Use BCEWithLogitsLoss which is AMP-safe and numerically more stable
    than sigmoid + BCELoss. The model's toxicity head must NOT apply
    sigmoid before passing here — sigmoid is applied at inference time only.
    """
    def __init__(
        self,
        n_tasks: int = 3,
        response_class_weights: Optional[torch.Tensor] = None,
        toxicity_pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.register_buffer("response_class_weights", response_class_weights)
        self.register_buffer("toxicity_pos_weight", toxicity_pos_weight)

    def forward(
        self,
        bioavail_pred: torch.Tensor,   # sigmoid output [B, 1]
        bioavail_true: torch.Tensor,   # float [B]
        response_pred: torch.Tensor,   # raw logits [B, 3]
        response_true: torch.Tensor,   # long [B]
        toxicity_logit: torch.Tensor,  # RAW LOGITS [B, 1]  (no sigmoid!)
        toxicity_true: torch.Tensor,   # float [B]
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if sample_weight is None:
            sample_weight = torch.ones_like(bioavail_true)
        sample_weight = sample_weight.float()
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)

        bio_per_sample = nn.MSELoss(reduction="none")(bioavail_pred.squeeze(), bioavail_true)
        loss_bioavail = (bio_per_sample * sample_weight).mean()

        response_per_sample = nn.CrossEntropyLoss(
            weight=self.response_class_weights,
            reduction="none",
        )(response_pred, response_true)
        loss_response = (response_per_sample * sample_weight).mean()

        # BCEWithLogitsLoss = sigmoid + BCE fused in one op → AMP-safe
        toxicity_per_sample = nn.BCEWithLogitsLoss(
            pos_weight=self.toxicity_pos_weight,
            reduction="none",
        )(toxicity_logit.squeeze(), toxicity_true)
        loss_toxicity = (toxicity_per_sample * sample_weight).mean()

        losses = [loss_bioavail, loss_response, loss_toxicity]

        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]

        return {
            "total":          total,
            "bioavailability": loss_bioavail,
            "response":        loss_response,
            "toxicity":        loss_toxicity,
            "task_weights":    torch.exp(-self.log_vars).detach(),
        }


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────

def compute_metrics(preds: Dict, batch: Dict) -> Dict[str, float]:
    """Compute all evaluation metrics for a batch."""
    from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

    metrics = {}

    # Bioavailability MAE
    bio_pred = preds["bioavailability"].squeeze().cpu().numpy()
    bio_true = batch["bioavailability"].cpu().numpy()
    metrics["bioavail_mae"] = float(mean_absolute_error(bio_true, bio_pred))

    # Response classification accuracy
    resp_pred = preds["response_class"].argmax(dim=-1).cpu().numpy()
    resp_true = batch["response_class"].cpu().numpy()
    metrics["response_acc"] = float(accuracy_score(resp_true, resp_pred))

    # Response AUROC (one-vs-rest for multiclass)
    resp_probs = torch.softmax(preds["response_class"], dim=-1).cpu().numpy()
    try:
        metrics["response_auroc"] = float(roc_auc_score(resp_true, resp_probs, multi_class="ovr"))
    except Exception:
        metrics["response_auroc"] = 0.5

    # Toxicity AUROC
    tox_pred = preds["toxicity"].squeeze().cpu().numpy()
    tox_true = batch["toxicity"].cpu().numpy()
    # Binarize toxicity at 0.5 for AUROC
    tox_binary = (tox_true > 0.5).astype(int)
    if len(np.unique(tox_binary)) > 1:
        metrics["toxicity_auroc"] = float(roc_auc_score(tox_binary, tox_pred))

    return metrics


# ─────────────────────────────────────────
# TRAINER CLASS
# ─────────────────────────────────────────

class Trainer:
    """
    Production trainer for MicroDrugNet.
    
    Features:
    - Mixed precision training (AMP)
    - Cosine annealing with warm restarts
    - Gradient clipping
    - Early stopping
    - Model checkpointing
    - Optional Weights & Biases logging
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        output_dir: str = "checkpoints",
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss function
        response_weights = config.get("response_class_weights")
        if response_weights is not None:
            response_weights = torch.tensor(response_weights, dtype=torch.float32, device=self.device)
        toxicity_pos_weight = config.get("toxicity_pos_weight")
        if toxicity_pos_weight is not None:
            toxicity_pos_weight = torch.tensor([toxicity_pos_weight], dtype=torch.float32, device=self.device)
        self.criterion = MultiTaskLoss(
            n_tasks=3,
            response_class_weights=response_weights,
            toxicity_pos_weight=toxicity_pos_weight,
        ).to(self.device)

        # Optimizer (AdamW with weight decay)
        all_params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
            betas=(0.9, 0.999)
        )

        # Scheduler: cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("T_0", 10),      # Restart every 10 epochs
            T_mult=config.get("T_mult", 2),  # Double period after each restart
            eta_min=config.get("lr_min", 1e-6)
        )

        # Mixed precision scaler (new API requires device string)
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Tracking
        self.best_val_auroc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = []

        # Optional W&B
        if use_wandb:
            try:
                import wandb
                wandb.init(project="MicroDrugNet", config=config)
                self.wandb = wandb
            except ImportError:
                print("W&B not installed. Skipping.")
                self.use_wandb = False

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_losses = {"total": 0, "bioavailability": 0, "response": 0, "toxicity": 0}
        all_metrics = []
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            microbiome = batch["microbiome"].to(self.device)
            drug_graph = batch["drug_graph"].to(self.device)
            bio_true = batch["bioavailability"].to(self.device)
            resp_true = batch["response_class"].to(self.device)
            tox_true = batch["toxicity"].to(self.device)
            condition_feat = batch["condition_feat"].to(self.device)
            drug_idx = batch["drug_idx"].to(self.device)
            sample_weight = batch["sample_weight"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            with autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(
                    drug_graph,
                    microbiome,
                    condition_feat=condition_feat,
                    drug_idx=drug_idx,
                )
                loss_dict = self.criterion(
                    preds["bioavailability"],         bio_true,
                    preds["response_class"],           resp_true,
                    preds["toxicity_logit"],           tox_true,
                    sample_weight=sample_weight,
                )

            # Backward pass
            self.scaler.scale(loss_dict["total"]).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track losses
            for k in ["total", "bioavailability", "response", "toxicity"]:
                total_losses[k] += loss_dict[k].item()
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()
        all_bio_preds, all_bio_true = [], []
        all_resp_preds, all_resp_true = [], []
        all_tox_preds, all_tox_true = [], []
        total_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            microbiome = batch["microbiome"].to(self.device)
            drug_graph = batch["drug_graph"].to(self.device)
            bio_true = batch["bioavailability"].to(self.device)
            resp_true = batch["response_class"].to(self.device)
            tox_true = batch["toxicity"].to(self.device)
            condition_feat = batch["condition_feat"].to(self.device)
            drug_idx = batch["drug_idx"].to(self.device)
            sample_weight = batch["sample_weight"].to(self.device)

            preds = self.model(
                drug_graph,
                microbiome,
                condition_feat=condition_feat,
                drug_idx=drug_idx,
            )
            loss_dict = self.criterion(
                preds["bioavailability"],         bio_true,
                preds["response_class"],           resp_true,
                preds["toxicity_logit"],           tox_true,
                sample_weight=sample_weight,
            )

            total_loss += loss_dict["total"].item()
            all_bio_preds.append(preds["bioavailability"].cpu().numpy())
            all_bio_true.append(bio_true.cpu().numpy())
            all_resp_preds.append(torch.softmax(preds["response_class"], dim=-1).cpu().numpy())
            all_resp_true.append(resp_true.cpu().numpy())
            all_tox_preds.append(preds["toxicity"].cpu().numpy())
            all_tox_true.append(tox_true.cpu().numpy())
            n_batches += 1

        # Aggregate metrics
        from sklearn.metrics import roc_auc_score, mean_absolute_error
        
        bio_pred_all = np.concatenate(all_bio_preds).squeeze()
        bio_true_all = np.concatenate(all_bio_true)
        resp_pred_all = np.concatenate(all_resp_preds)
        resp_true_all = np.concatenate(all_resp_true)
        tox_pred_all = np.concatenate(all_tox_preds).squeeze()
        tox_true_all = np.concatenate(all_tox_true)

        metrics = {
            "val_loss": total_loss / n_batches,
            "val_bioavail_mae": float(mean_absolute_error(bio_true_all, bio_pred_all)),
            "val_response_acc": float((resp_pred_all.argmax(1) == resp_true_all).mean()),
        }

        try:
            metrics["val_response_auroc"] = float(
                roc_auc_score(resp_true_all, resp_pred_all, multi_class="ovr")
            )
        except Exception:
            metrics["val_response_auroc"] = 0.5

        tox_binary = (tox_true_all > 0.5).astype(int)
        if len(np.unique(tox_binary)) > 1:
            metrics["val_toxicity_auroc"] = float(roc_auc_score(tox_binary, tox_pred_all))

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (AUROC: {metrics.get('val_response_auroc', 0):.4f})")

    def fit(self, n_epochs: int = 100, patience: int = 15) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"  MicroDrugNet Training")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {n_epochs} | Patience: {patience}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Print epoch summary
            auroc = val_metrics.get("val_response_auroc", 0)
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {train_losses['total']:.4f} | "
                f"Val AUROC: {auroc:.4f} | "
                f"Val Acc: {val_metrics['val_response_acc']:.3f} | "
                f"Bioavail MAE: {val_metrics['val_bioavail_mae']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Track history
            record = {"epoch": epoch, **train_losses, **val_metrics, "lr": current_lr}
            self.history.append(record)

            # W&B logging
            if self.use_wandb:
                self.wandb.log(record)

            # Check for improvement
            is_best = auroc > self.best_val_auroc
            if is_best:
                self.best_val_auroc = auroc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint every 5 epochs + best
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best Val AUROC: {self.best_val_auroc:.4f}")
        print(f"  History saved to: {history_path}")
        print(f"{'='*60}")

        return {"best_epoch": self.best_epoch, "best_val_auroc": self.best_val_auroc}


# ─────────────────────────────────────────
# MAIN TRAINING ENTRYPOINT
# ─────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "n_taxa": 500,
    "n_samples": 30000,
    "data_path": "data/processed/training_dataset.csv",
    "batch_size": 32,
    "num_workers": 0,
    "val_split": 0.15,
    "test_split": 0.15,
    
    # Model
    "drug_dim": 320,
    "n_genes": 512,
    "use_patient": False,
    "dropout": 0.15,
    
    # Training
    "n_epochs": 200,
    "lr": 7e-5,
    "lr_min": 1e-6,
    "weight_decay": 1e-5,
    "T_0": 20,
    "T_mult": 2,
    "patience": 20,
    "seed": 42,
}


def main(config: Dict = None, use_wandb: bool = False):
    if config is None:
        config = DEFAULT_CONFIG

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ── 1. Generate/Load Data ──
    print("Preparing dataset...")
    from microdrug.data_utils import generate_synthetic_dataset, get_dataloaders

    data_path = config.get("data_path")
    if data_path and Path(data_path).exists():
        print(f"Loading real dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Real training CSV not found. Falling back to synthetic dataset.")
        df = generate_synthetic_dataset(
            n_samples=config["n_samples"],
            n_taxa=config["n_taxa"],
            seed=config["seed"]
        )

    train_loader, val_loader, test_loader = get_dataloaders(
        df,
        n_taxa=config["n_taxa"],
        batch_size=config["batch_size"],
        val_frac=config["val_split"],
        test_frac=config["test_split"],
        seed=config["seed"],
        num_workers=config.get("num_workers", 0),
    )
    n_drugs = int(df["drug_name"].astype(str).str.lower().str.strip().nunique())

    train_df = train_loader.dataset.df
    class_counts = train_df["response_class"].value_counts().sort_index()
    class_weights = (len(train_df) / (len(class_counts) * class_counts)).tolist()
    tox_binary = (train_df["toxicity"] > 0.5).astype(int)
    n_pos = int(tox_binary.sum())
    n_neg = int(len(tox_binary) - n_pos)
    tox_pos_weight = float(n_neg / max(n_pos, 1))
    config["response_class_weights"] = class_weights
    config["toxicity_pos_weight"] = tox_pos_weight
    print(f"Response class counts (train): {class_counts.to_dict()}")
    print(f"Response class weights: {[round(x, 3) for x in class_weights]}")
    print(f"Toxicity positives (train): {n_pos}/{len(train_df)} | pos_weight={tox_pos_weight:.2f}")

    # ── 2. Build Model ──
    from microdrug.model import MicroDrugNet
    
    model = MicroDrugNet(
        n_taxa=config["n_taxa"],
        n_genes=config["n_genes"],
        drug_dim=config["drug_dim"],
        use_patient=config["use_patient"],
        dropout=config["dropout"],
        n_drugs=n_drugs,
    )

    # ── 3. Train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config.get("output_dir", "checkpoints"),
        use_wandb=use_wandb,
    )

    results = trainer.fit(n_epochs=config["n_epochs"], patience=config["patience"])

    # ── 4. Final Test Evaluation ──
    print("\nRunning test set evaluation...")
    best_model_path = Path(config.get("output_dir", "checkpoints")) / "best_model.pt"
    best_ckpt = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    
    # Run test evaluation using validate() on test_loader
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save test results
    with open(Path(config.get("output_dir", "checkpoints")) / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return model, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MicroDrugNet")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_samples", type=int, default=5000, help="Synthetic dataset size")
    parser.add_argument("--data_path", type=str, default="data/processed/training_dataset.csv")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in vars(args).items() if k in config})

    main(config=config, use_wandb=args.wandb)
