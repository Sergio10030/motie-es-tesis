"""C5: Brazo B - Fine-tuning con encoders preentrenados en español."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("motie_es.arm_b")


class IncidentDataset(Dataset):
    """Dataset PyTorch para reportes de incidencia."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class ArmBFinetuning:
    """Brazo B: fine-tuning de encoders en español.

    Modelos soportados: BETO, RoBERTuito, BERTuit, RoBERTa-BNE.
    Configuración: lr=2e-5, batch=16, weight_decay=0.01, 6 épocas,
    ponderación de clases inversamente proporcional a la frecuencia.
    """

    SUPPORTED_MODELS = {
        "beto": "dccuchile/bert-base-spanish-wwm-uncased",
        "robertuito": "pysentimiento/robertuito-base-uncased",
        "bertuit": "BSC-LT/bertuit-base-cased",
        "roberta_bne": "BSC-LT/roberta-base-bne",
    }

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-base-uncased",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        weight_decay: float = 0.01,
        num_epochs: int = 6,
        max_length: int = 512,
        warmup_ratio: float = 0.1,
        class_weighting: bool = True,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 3,
        device: Optional[str] = None,
        seed: int = 42,
        checkpoint_dir: str = "models/checkpoints",
    ) -> None:
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.class_weighting = class_weighting
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir)
        set_seed(seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer: Optional[Any] = None
        self.model: Optional[nn.Module] = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.history: List[Dict[str, float]] = []

    def _compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """w_c = N / (K * n_c)"""
        arr = np.array(labels)
        classes = np.unique(arr)
        N = len(arr)
        K = len(classes)
        weights = []
        for c in classes:
            n_c = int(np.sum(arr == c))
            weights.append(N / (K * n_c) if n_c > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float).to(self.device)

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
    ) -> "ArmBFinetuning":
        """Entrena el modelo con fine-tuning."""
        logger.info("Entrenando Brazo B con modelo %s", self.model_name)
        t0 = time.time()

        # Mapeo de etiquetas
        unique_labels = sorted(set(train_labels))
        self.label2id = {lab: i for i, lab in enumerate(unique_labels)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}
        num_labels = len(unique_labels)

        # Tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)

        train_enc = [self.label2id[l] for l in train_labels]
        train_ds = IncidentDataset(train_texts, train_enc, self.tokenizer, self.max_length)

        class_weights = None
        if self.class_weighting:
            class_weights = self._compute_class_weights(train_enc)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = -1.0
        patience_counter = 0
        self.history = []

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            metrics = {"epoch": epoch, "loss": avg_loss}

            if val_texts and val_labels:
                val_metrics = self.evaluate(val_texts, val_labels)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                val_f1 = val_metrics.get("f1_macro", 0.0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    patience_counter += 1

            self.history.append(metrics)
            logger.info("Época %d/%d | loss=%.4f", epoch, self.num_epochs, avg_loss)

            if patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping en época %d", epoch)
                break

        logger.info("Brazo B entrenado en %.2f s", time.time() - t0)
        return self

    def _save_checkpoint(self, name: str) -> None:
        """Guarda checkpoint del modelo."""
        path = self.checkpoint_dir / f"{self.model_name.replace('/', '_')}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        logger.info("Checkpoint guardado en %s", path)

    def predict(self, texts: List[str]) -> Dict[str, List[str]]:
        """Predice etiquetas."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("El modelo no está entrenado.")
        self.model.eval()
        ds = IncidentDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(ds, batch_size=self.batch_size)
        preds: List[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        return {"tipo_predicho": [self.id2label[p] for p in preds]}

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Evalúa el modelo con F1 macro."""
        from sklearn.metrics import f1_score
        preds = self.predict(texts)["tipo_predicho"]
        return {
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }