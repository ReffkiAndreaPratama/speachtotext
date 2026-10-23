"""
src/model.py
============
Arsitektur Deep Learning untuk Speech-to-Text Bahasa Bengkulu.

Arsitektur: CNN-BiLSTM-CTC
==========================================
Pipeline forward pass:
    Input  : Log-Mel Spectrogram  [B, n_mels, T]
    ↓
    CNN Encoder   : beberapa blok Conv2D + BatchNorm + ReLU
                    untuk menangkap pola lokal frekuensi-waktu
    ↓
    Reshape        : [B, C*freq, T'] → [T', B, features]
    ↓
    BiLSTM         : model temporal jangka panjang (bidirectional)
    ↓
    Projection     : Linear → vocab_size
    ↓
    Output : log-softmax logits [T', B, vocab_size]
             digunakan bersama CTCLoss

Mengapa CNN-BiLSTM-CTC?
- Sesuai dengan literatur ASR low-resource (minim data).
- CTC tidak memerlukan alignment manual antara audio dan transkripsi.
- Dapat berjalan di CPU maupun GPU.
- Kompleksitas terkontrol, cocok untuk dataset skala kecil-menengah.

Kelas:
    - ModelConfig           : dataclass konfigurasi hyperparameter
    - ConvBlock             : blok konvolusi CNN
    - CNNEncoder            : encoder CNN multi-layer
    - SpeechRecognitionModel: model lengkap CNN-BiLSTM-CTC
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# Model Config
# ===========================================================================

@dataclass
class ModelConfig:
    """
    Konfigurasi hyperparameter model.

    Attributes
    ----------
    vocab_size      : jumlah token (karakter + spesial token)
    n_mels          : jumlah Mel filter bank (harus sama dengan preprocessor)
    cnn_channels    : list jumlah channel untuk setiap blok CNN
    cnn_kernel_size : ukuran kernel konvolusi (height x width)
    cnn_stride      : stride konvolusi
    lstm_hidden     : ukuran hidden state BiLSTM (per direction)
    lstm_layers     : jumlah layer BiLSTM
    lstm_dropout    : dropout antar layer BiLSTM (0 jika lstm_layers=1)
    fc_dropout      : dropout sebelum layer proyeksi akhir
    """

    vocab_size: int = 64          # diperbarui saat tokenizer dibangun
    n_mels: int = 80
    cnn_channels: list = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: Tuple[int, int] = (3, 3)
    cnn_stride: Tuple[int, int] = (1, 1)
    lstm_hidden: int = 256
    lstm_layers: int = 3
    lstm_dropout: float = 0.3
    fc_dropout: float = 0.3

    def __post_init__(self):
        # Validasi
        if self.vocab_size < 5:
            raise ValueError("vocab_size terlalu kecil (minimal 5 token spesial).")
        if self.n_mels < 16:
            raise ValueError("n_mels terlalu kecil.")
        if not self.cnn_channels:
            raise ValueError("cnn_channels tidak boleh kosong.")
        if self.lstm_layers == 1 and self.lstm_dropout > 0:
            # PyTorch memperingatkan dropout pada single-layer LSTM
            self.lstm_dropout = 0.0

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_mels": self.n_mels,
            "cnn_channels": self.cnn_channels,
            "cnn_kernel_size": list(self.cnn_kernel_size),
            "cnn_stride": list(self.cnn_stride),
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "fc_dropout": self.fc_dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        if "cnn_kernel_size" in d:
            d["cnn_kernel_size"] = tuple(d["cnn_kernel_size"])
        if "cnn_stride" in d:
            d["cnn_stride"] = tuple(d["cnn_stride"])
        return cls(**d)


# ===========================================================================
# Blok CNN
# ===========================================================================

class ConvBlock(nn.Module):
    """
    Blok konvolusi: Conv2d → BatchNorm2d → ReLU → MaxPool2d (opsional).

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : tuple (kH, kW)
    stride       : tuple (sH, sW)
    pool         : bool — terapkan MaxPool2d(2, 2) setelah ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        pool: bool = False,
    ):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, freq, time]
        x = self.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


# ===========================================================================
# CNN Encoder
# ===========================================================================

class CNNEncoder(nn.Module):
    """
    Encoder CNN multi-layer.

    Menerima Log-Mel Spectrogram [B, 1, n_mels, T] dan menghasilkan
    representasi fitur [B, C_last * freq_reduced, T_reduced].

    MaxPool hanya diterapkan pada blok pertama (dimensi frekuensi) agar
    dimensi waktu tidak tereduksi terlalu banyak, sehingga alignment
    CTC tetap bermakna.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = [1] + config.cnn_channels  # channel pertama = 1 (grayscale)
        blocks = []
        for i in range(len(config.cnn_channels)):
            pool = (i == 0)  # hanya blok pertama yang pakai MaxPool
            blocks.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=config.cnn_kernel_size,
                    stride=config.cnn_stride,
                    pool=pool,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.config = config

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        x : [B, n_mels, T]

        Returns
        -------
        out     : [B, C * freq_reduced, T_reduced]
        time_out: int — T_reduced (untuk menghitung panjang sekuens CTC)
        """
        # Tambah channel dimension: [B, 1, n_mels, T]
        x = x.unsqueeze(1)
        x = self.blocks(x)               # [B, C_last, freq_reduced, T_reduced]

        B, C, F, T = x.shape
        # Gabung dimensi channel dan frekuensi → [B, C*F, T]
        x = x.reshape(B, C * F, T)
        return x, T

    def compute_output_freq(self, n_mels: int) -> int:
        """Hitung dimensi frekuensi setelah CNN (untuk menentukan LSTM input size)."""
        dummy = torch.zeros(1, n_mels, 100)
        with torch.no_grad():
            out, _ = self.forward(dummy)
        return out.shape[1]


# ===========================================================================
# Model Utama: CNN-BiLSTM-CTC
# ===========================================================================

class SpeechRecognitionModel(nn.Module):
    """
    Model Speech Recognition CNN-BiLSTM-CTC untuk Bahasa Bengkulu.

    Aliran data:
        [B, n_mels, T]
        → CNN Encoder   → [B, cnn_out_features, T']
        → Permute       → [T', B, cnn_out_features]
        → BiLSTM        → [T', B, lstm_hidden * 2]
        → Dropout
        → Linear        → [T', B, vocab_size]
        → Log Softmax   → [T', B, vocab_size]

    Loss: nn.CTCLoss(blank=0, zero_infinity=True)

    Parameters
    ----------
    config : ModelConfig
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # CNN Encoder
        self.encoder = CNNEncoder(config)

        # Hitung ukuran output CNN → input BiLSTM
        cnn_out_size = self.encoder.compute_output_freq(config.n_mels)
        logger.info("CNN output size (features per time step): %d", cnn_out_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=False,        # input [T, B, features]
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )

        # Proyeksi ke vocab
        self.dropout = nn.Dropout(p=config.fc_dropout)
        self.fc = nn.Linear(config.lstm_hidden * 2, config.vocab_size)

        # Inisialisasi bobot
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SpeechRecognitionModel diinisialisasi: %d parameter trainable",
            total_params,
        )

    def _init_weights(self) -> None:
        """Inisialisasi bobot dengan Xavier/Orthogonal."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias = 1 untuk LSTM (teknik umum)
                if "bias_hh" in name or "bias_ih" in name:
                    n = param.size(0)
                    start = n // 4
                    end = n // 2
                    param.data[start:end].fill_(1.0)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        features      : [B, n_mels, T]
        input_lengths : [B] — panjang time frames per sampel (sebelum padding)
                        Jika None, diasumsikan semua penuh (T_full).

        Returns
        -------
        log_probs     : [T', B, vocab_size]  — log softmax output
        output_lengths: [B]                  — panjang T' per sampel setelah CNN
        """
        # 1) CNN Encoder
        cnn_out, T_out = self.encoder(features)   # [B, features, T']
        B = cnn_out.shape[0]

        # 2) Hitung output lengths (diperlukan CTCLoss)
        if input_lengths is not None:
            output_lengths = self._compute_output_lengths(input_lengths, T_out, features.shape[-1])
        else:
            output_lengths = torch.full((B,), T_out, dtype=torch.long)

        # 3) [B, features, T'] → [T', B, features]
        cnn_out = cnn_out.permute(2, 0, 1)        # [T', B, features]

        # 4) BiLSTM
        lstm_out, _ = self.lstm(cnn_out)           # [T', B, lstm_hidden*2]

        # 5) Dropout + Linear + Log Softmax
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)                 # [T', B, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # [T', B, vocab_size]

        return log_probs, output_lengths

    def _compute_output_lengths(
        self,
        input_lengths: torch.Tensor,
        T_out: int,
        T_in: int,
    ) -> torch.Tensor:
        """
        Hitung panjang output CTC secara proporsional terhadap
        reduksi temporal oleh CNN.
        """
        scale = T_out / max(T_in, 1)
        output_lengths = (input_lengths.float() * scale).long()
        # Klem agar tidak melebihi T_out dan tidak kurang dari 1
        output_lengths = output_lengths.clamp(min=1, max=T_out)
        return output_lengths

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Kembalikan jumlah parameter model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Simpan / Load
    # ------------------------------------------------------------------

    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        """
        Simpan state_dict dan konfigurasi model.

        Format file yang disimpan:
        {
            "model_state_dict": ...,
            "config": {...},
            "extra": {...}   # epoch, loss, dll.
        }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        logger.info("Model disimpan ke %s", path)

    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> "SpeechRecognitionModel":
        """
        Load model dari file checkpoint.

        Parameters
        ----------
        path   : Path ke file .pt/.pth
        device : torch.device — default: cpu

        Returns
        -------
        model  : SpeechRecognitionModel (eval mode)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {path}")

        if device is None:
            device = torch.device("cpu")

        payload = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()

        extra = payload.get("extra", {})
        logger.info(
            "Model di-load dari %s | device=%s | extra=%s",
            path, device, extra,
        )
        return model, extra

    def __repr__(self) -> str:
        n_params = self.get_num_parameters()
        return (
            f"SpeechRecognitionModel("
            f"vocab={self.config.vocab_size}, "
            f"n_mels={self.config.n_mels}, "
            f"cnn={self.config.cnn_channels}, "
            f"lstm={self.config.lstm_hidden}x{self.config.lstm_layers}, "
            f"params={n_params:,})"
        )
