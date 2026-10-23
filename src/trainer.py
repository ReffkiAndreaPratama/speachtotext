"""
src/trainer.py
==============
Training loop untuk Speech-to-Text Bahasa Bengkulu.

Fitur:
    - CTC Loss dengan blank token = 0
    - Adam optimizer dengan learning rate scheduler (ReduceLROnPlateau)
    - Checkpoint per epoch (best model berdasarkan validation loss)
    - Early stopping jika val_loss tidak membaik selama N epoch
    - Logging per batch dan per epoch ke file + console
    - Reproducibility melalui seed
    - Dukungan GPU (CUDA) dengan fallback ke CPU
    - Training history disimpan ke results/training_history.json
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.model import SpeechRecognitionModel, ModelConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# Training Config
# ===========================================================================

@dataclass
class TrainingConfig:
    """
    Hyperparameter dan konfigurasi training.

    Attributes
    ----------
    batch_size      : ukuran batch
    learning_rate   : learning rate awal Adam
    epochs          : jumlah epoch maksimum
    sample_rate     : sample rate audio (Hz)
    max_audio_length: durasi maksimum audio (detik)
    seed            : random seed
    early_stop_patience : jumlah epoch tanpa improvement sebelum stop
    lr_patience     : jumlah epoch sebelum LR diturunkan
    lr_factor       : faktor pengurangan LR (new_lr = lr * factor)
    grad_clip       : gradient clipping maksimum (0 = tidak aktif)
    log_interval    : setiap N batch cetak log
    checkpoint_dir  : direktori penyimpanan checkpoint
    results_dir     : direktori hasil (history, dll.)
    """

    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 50
    sample_rate: int = 16000
    max_audio_length: float = 10.0
    seed: int = 42
    early_stop_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    grad_clip: float = 5.0
    log_interval: int = 10
    checkpoint_dir: str = "models"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """
    Mengelola siklus training, validasi, checkpointing, dan early stopping.

    Parameters
    ----------
    model        : SpeechRecognitionModel
    config       : TrainingConfig
    device       : torch.device — ditentukan otomatis jika None
    """

    BEST_MODEL_FILE = "best_model.pt"
    LAST_CHECKPOINT_FILE = "last_checkpoint.pt"
    HISTORY_FILE = "training_history.json"
    VOCAB_FILE = "vocab.txt"

    def __init__(
        self,
        model: SpeechRecognitionModel,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config

        # Tentukan device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,
        )

        # Scheduler: turunkan LR saat val_loss stagnan
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=1e-6,
        )

        # CTC Loss — blank = 0 (PAD token)
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Paths
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.results_dir = Path(config.results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time_sec": [],
        }
        self.best_val_loss = math.inf
        self.epochs_no_improve = 0
        self.start_epoch = 0

        logger.info(
            "Trainer diinisialisasi | device=%s | optimizer=Adam(lr=%.1e) | "
            "epochs=%d | early_stop=%d",
            self.device, config.learning_rate, config.epochs, config.early_stop_patience,
        )

    # ------------------------------------------------------------------
    # Training loop utama
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[Path] = None,
    ) -> Dict:
        """
        Jalankan training penuh.

        Parameters
        ----------
        train_loader : DataLoader training
        val_loader   : DataLoader validasi
        resume_from  : Path ke checkpoint untuk melanjutkan training

        Returns
        -------
        history : dict — riwayat loss dan learning rate per epoch
        """
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        logger.info("=" * 60)
        logger.info("Mulai training | epoch %d → %d", self.start_epoch + 1, self.config.epochs)
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_num = epoch + 1
            t0 = time.time()

            # --- Training ---
            train_loss = self._train_one_epoch(train_loader, epoch_num)

            # --- Validasi ---
            val_loss = self._validate(val_loader, epoch_num)

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Catat history
            self.history["train_loss"].append(round(train_loss, 6))
            self.history["val_loss"].append(round(val_loss, 6))
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time_sec"].append(round(elapsed, 2))

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "lr=%.2e | waktu=%.1fs",
                epoch_num, self.config.epochs,
                train_loss, val_loss, current_lr, elapsed,
            )

            # --- Simpan checkpoint terakhir ---
            self._save_checkpoint(epoch_num, val_loss, is_best=False)

            # --- Simpan best model ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch_num, val_loss, is_best=True)
                logger.info("  ✓ Best model diperbarui (val_loss=%.4f)", val_loss)
            else:
                self.epochs_no_improve += 1
                logger.info(
                    "  ✗ Tidak ada peningkatan (%d/%d sebelum early stop)",
                    self.epochs_no_improve, self.config.early_stop_patience,
                )

            # --- Scheduler ---
            self.scheduler.step(val_loss)

            # --- Early stopping ---
            if self.epochs_no_improve >= self.config.early_stop_patience:
                logger.info(
                    "Early stopping aktif pada epoch %d (tidak ada peningkatan "
                    "selama %d epoch).",
                    epoch_num, self.config.early_stop_patience,
                )
                break

        # Simpan history
        self._save_history()
        logger.info("Training selesai. Best val_loss=%.4f", self.best_val_loss)
        return self.history

    # ------------------------------------------------------------------
    # Satu epoch training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Jalankan satu epoch training. Kembalikan rata-rata loss."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        skipped = 0

        for batch_idx, batch in enumerate(loader):
            features, targets, input_lengths, target_lengths = batch

            # Pindah ke device
            features = features.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            try:
                log_probs, output_lengths = self.model(features, input_lengths)
            except Exception as exc:
                logger.warning("Forward pass gagal pada batch %d: %s", batch_idx, exc)
                skipped += 1
                continue

            # CTC Loss
            # log_probs: [T', B, vocab] — CTCLoss butuh [T, B, C]
            # targets  : [B, S]         — CTCLoss butuh [sum(target_lengths)]
            targets_flat = targets.view(-1)[:target_lengths.sum().item()]
            # Flatten targets sesuai panjang sebenarnya
            targets_ctc = []
            for i, tl in enumerate(target_lengths):
                targets_ctc.append(targets[i, :tl.item()])
            targets_ctc = torch.cat(targets_ctc)

            try:
                loss = self.criterion(
                    log_probs,
                    targets_ctc,
                    output_lengths,
                    target_lengths,
                )
            except Exception as exc:
                logger.warning("CTC Loss gagal pada batch %d: %s", batch_idx, exc)
                skipped += 1
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Loss NaN/Inf pada batch %d, skip.", batch_idx)
                skipped += 1
                continue

            # Backward + gradient clip + update
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    "  Epoch %d | Batch %d/%d | loss=%.4f",
                    epoch, batch_idx + 1, len(loader), loss.item(),
                )

        if skipped > 0:
            logger.warning("  %d batch dilewati karena error.", skipped)

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    # ------------------------------------------------------------------
    # Validasi
    # ------------------------------------------------------------------

    def _validate(self, loader: DataLoader, epoch: int) -> float:
        """Jalankan validasi. Kembalikan rata-rata loss."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in loader:
                features, targets, input_lengths, target_lengths = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)

                try:
                    log_probs, output_lengths = self.model(features, input_lengths)
                except Exception:
                    continue

                targets_ctc = []
                for i, tl in enumerate(target_lengths):
                    targets_ctc.append(targets[i, :tl.item()])
                targets_ctc = torch.cat(targets_ctc)

                try:
                    loss = self.criterion(
                        log_probs, targets_ctc, output_lengths, target_lengths
                    )
                except Exception:
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """Simpan checkpoint model."""
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "history": self.history,
            "training_config": self.config.to_dict(),
        }
        extra = {
            "epoch": epoch,
            "val_loss": round(val_loss, 6),
            "best_val_loss": round(self.best_val_loss, 6),
        }

        if is_best:
            path = self.checkpoint_dir / self.BEST_MODEL_FILE
            self.model.save(path, extra=extra)
            # Simpan juga state trainer (untuk resume)
            trainer_state_path = self.checkpoint_dir / "trainer_state_best.json"
            with open(trainer_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        else:
            path = self.checkpoint_dir / self.LAST_CHECKPOINT_FILE
            self.model.save(path, extra=extra)
            trainer_state_path = self.checkpoint_dir / "trainer_state_last.json"
            with open(trainer_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def _load_checkpoint(self, path: Path) -> None:
        """Load checkpoint untuk melanjutkan training."""
        path = Path(path)
        if not path.exists():
            logger.warning("Checkpoint tidak ditemukan: %s. Mulai dari awal.", path)
            return

        # Load model weights
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)

        # Load trainer state jika ada
        state_file = path.parent / f"trainer_state_{path.stem.split('_')[-1]}.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.start_epoch = state.get("epoch", 0)
            self.best_val_loss = state.get("best_val_loss", math.inf)
            self.epochs_no_improve = state.get("epochs_no_improve", 0)
            self.history = state.get("history", self.history)
            logger.info(
                "Resume dari epoch %d | best_val_loss=%.4f",
                self.start_epoch, self.best_val_loss,
            )
        else:
            extra = payload.get("extra", {})
            self.start_epoch = extra.get("epoch", 0)
            self.best_val_loss = extra.get("best_val_loss", math.inf)

    def _save_history(self) -> None:
        """Simpan training history ke JSON."""
        path = self.results_dir / self.HISTORY_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": self.history,
                    "best_val_loss": self.best_val_loss,
                    "config": self.config.to_dict(),
                },
                f,
                indent=2,
            )
        logger.info("Training history disimpan ke %s", path)

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seed untuk reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Random seed = %d", seed)

    def get_device_info(self) -> str:
        """Kembalikan info device yang digunakan."""
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            mem_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            return f"CUDA — {name} ({mem_gb:.1f} GB)"
        return "CPU"
