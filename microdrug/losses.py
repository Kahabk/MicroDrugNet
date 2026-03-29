"""
MicroDrugNet — Multi-task Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroDrugLoss(nn.Module):
    """
    Weighted combination of:
      • MSE on bioavailability score
      • CrossEntropy on response class
      • Cosine embedding loss on metabolite fingerprint (optional)
    """
    def __init__(self, w_bio=1.0, w_resp=0.5, w_meta=0.3):
        super().__init__()
        self.w_bio  = w_bio
        self.w_resp = w_resp
        self.w_meta = w_meta

    @staticmethod
    def _get_response_logits(preds):
        if "response_logits" in preds:
            return preds["response_logits"]
        if "response_class" in preds:
            return preds["response_class"]
        raise KeyError("Expected preds to contain 'response_logits' or 'response_class'")

    @staticmethod
    def _get_response_labels(labels):
        if "response" in labels:
            return labels["response"]
        if "response_class" in labels:
            return labels["response_class"]
        raise KeyError("Expected labels to contain 'response' or 'response_class'")

    def forward(self, preds, labels, metabolite_targets=None):
        loss_bio  = F.mse_loss(preds["bioavailability"], labels["bioavailability"])
        loss_resp = F.cross_entropy(
            self._get_response_logits(preds),
            self._get_response_labels(labels),
        )

        loss = self.w_bio * loss_bio + self.w_resp * loss_resp

        if metabolite_targets is not None and self.w_meta > 0:
            loss_meta = 1 - F.cosine_similarity(
                preds["metabolites"], metabolite_targets, dim=-1
            ).mean()
            loss = loss + self.w_meta * loss_meta

        return loss, {"bio": loss_bio.item(), "resp": loss_resp.item()}
