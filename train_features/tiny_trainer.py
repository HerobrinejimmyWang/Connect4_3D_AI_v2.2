from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TinyTrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    grad_clip_norm: float = 5.0
    policy_loss_weight: float = 1.0
    teacher_kl_weight: float = 0.25
    value_loss_weight: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TinyPolicyTrainer:
    def __init__(self, model: torch.nn.Module, config: TinyTrainConfig | None = None) -> None:
        self.model = model
        self.config = config or TinyTrainConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(1, int(self.config.epochs) + 1):
            train_metrics = self.train_one_epoch(train_loader)
            train_metrics["epoch"] = float(epoch)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                for key, value in val_metrics.items():
                    train_metrics[f"val_{key}"] = float(value)

            # Print only epoch number and training loss to terminal
            loss = train_metrics.get("loss")
            if loss is None:
                print(f"epoch {epoch} loss N/A")
            else:
                print(f"epoch {epoch} loss {loss:.6f}")

            history.append(train_metrics)
        return history

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_value = 0.0
        total_weight = 0.0
        total_correct = 0.0
        total_samples = 0.0

        for batch in loader:
            global_features = batch["global_features"].to(self.device)
            candidate_features = batch["candidate_features"].to(self.device)
            valid_mask = batch["valid_mask"].to(self.device)
            target_idx = batch["target_idx"].to(self.device)
            sample_weight = batch["sample_weight"].to(self.device)
            teacher_probs = batch["teacher_probs"].to(self.device)
            teacher_valid = batch["teacher_valid"].to(self.device)
            teacher_value = batch["teacher_value"].to(self.device)
            teacher_value_valid = batch["teacher_value_valid"].to(self.device)

            output = self.model(global_features, candidate_features, valid_mask=valid_mask, return_value=True)
            if isinstance(output, tuple):
                logits, pred_value = output
            else:
                logits = output
                pred_value = None
            ce_per_sample = F.cross_entropy(logits, target_idx, reduction="none")
            weighted_ce = (ce_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)

            kl_loss = self._teacher_kl_loss(logits, teacher_probs, teacher_valid, sample_weight)
            value_loss = self._teacher_value_loss(pred_value, teacher_value, teacher_value_valid, sample_weight)

            total = (
                float(self.config.policy_loss_weight) * weighted_ce
                + float(self.config.teacher_kl_weight) * kl_loss
                + float(self.config.value_loss_weight) * value_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.grad_clip_norm))
            self.optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == target_idx).float()
                total_correct += float(torch.sum(correct).item())
                total_samples += float(target_idx.shape[0])

                batch_weight = float(sample_weight.sum().item())
                total_weight += batch_weight
                total_loss += float(total.item()) * batch_weight
                total_ce += float(weighted_ce.item()) * batch_weight
                total_kl += float(kl_loss.item()) * batch_weight
                total_value += float(value_loss.item()) * batch_weight

        denom = max(1e-6, total_weight)
        return {
            "loss": total_loss / denom,
            "policy_ce": total_ce / denom,
            "teacher_kl": total_kl / denom,
            "teacher_value": total_value / denom,
            "acc": total_correct / max(1.0, total_samples),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_value = 0.0
        total_weight = 0.0
        total_correct = 0.0
        total_samples = 0.0

        with torch.no_grad():
            for batch in loader:
                global_features = batch["global_features"].to(self.device)
                candidate_features = batch["candidate_features"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)
                target_idx = batch["target_idx"].to(self.device)
                sample_weight = batch["sample_weight"].to(self.device)
                teacher_value = batch["teacher_value"].to(self.device)
                teacher_value_valid = batch["teacher_value_valid"].to(self.device)

                output = self.model(global_features, candidate_features, valid_mask=valid_mask, return_value=True)
                if isinstance(output, tuple):
                    logits, pred_value = output
                else:
                    logits = output
                    pred_value = None
                ce_per_sample = F.cross_entropy(logits, target_idx, reduction="none")
                weighted_ce = (ce_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
                value_loss = self._teacher_value_loss(pred_value, teacher_value, teacher_value_valid, sample_weight)

                pred = torch.argmax(logits, dim=-1)
                total_correct += float(torch.sum((pred == target_idx).float()).item())
                total_samples += float(target_idx.shape[0])

                batch_weight = float(sample_weight.sum().item())
                total_weight += batch_weight
                total_loss += float(weighted_ce.item()) * batch_weight
                total_value += float(value_loss.item()) * batch_weight

        return {
            "loss": total_loss / max(1e-6, total_weight),
            "teacher_value": total_value / max(1e-6, total_weight),
            "acc": total_correct / max(1.0, total_samples),
        }

    def _teacher_kl_loss(
        self,
        logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_valid: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        valid_rows = teacher_valid > 0.5
        if not torch.any(valid_rows):
            return torch.zeros((), device=logits.device)

        selected_logits = logits[valid_rows]
        selected_teacher = teacher_probs[valid_rows]
        selected_weight = sample_weight[valid_rows]

        selected_teacher = torch.clamp(selected_teacher, min=0.0)
        selected_teacher = selected_teacher / selected_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        log_probs = F.log_softmax(selected_logits, dim=-1)
        kl_per_sample = torch.sum(
            selected_teacher * (torch.log(selected_teacher + 1e-8) - log_probs),
            dim=-1,
        )
        return (kl_per_sample * selected_weight).sum() / selected_weight.sum().clamp_min(1e-6)

    def _teacher_value_loss(
        self,
        pred_value: torch.Tensor | None,
        teacher_value: torch.Tensor,
        teacher_value_valid: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        if pred_value is None:
            return torch.zeros((), device=teacher_value.device)

        valid_rows = teacher_value_valid > 0.5
        if not torch.any(valid_rows):
            return torch.zeros((), device=teacher_value.device)

        selected_pred = pred_value[valid_rows]
        selected_target = teacher_value[valid_rows]
        selected_weight = sample_weight[valid_rows]

        mse = (selected_pred - selected_target) ** 2
        return torch.sum(mse * selected_weight) / selected_weight.sum().clamp_min(1e-6)

    def export_config(self) -> dict:
        return asdict(self.config)
