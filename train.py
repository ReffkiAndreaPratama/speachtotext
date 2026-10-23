"""
train.py
========
Script utama training Speech-to-Text Bahasa Bengkulu.

MODE 1 — Training dengan dataset sendiri:
    python train.py

Dengan parameter kustom:
    python train.py \
        --batch_size 16 \
        --learning_rate 1e-3 \
        --epochs 100 \
        --sample_rate 16000 \
        --max_audio_length 10.0 \
        --seed 42

Melanjutkan training dari checkpoint:
    python train.py --resume models/last_checkpoint.pt

Dataset speaker-independent split:
    python train.py --speaker_independent

Lihat semua opsi:
    python train.py --help

Persyaratan:
    - dataset/metadata.csv sudah disiapkan (jalankan prepare_dataset.py dulu)
    - dataset/audio/ berisi file audio
    - Python 3.11+, PyTorch, torchaudio
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging ke console + file
LOG_DIR = Path("results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training Speech-to-Text Bahasa Bengkulu (CNN-BiLSTM-CTC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--dataset_dir", type=Path, default=Path("dataset"),
        help="Direktori dataset (berisi audio/ dan metadata.csv)",
    )
    dataset_group.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Proporsi data training",
    )
    dataset_group.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Proporsi data validasi",
    )
    dataset_group.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Proporsi data test",
    )
    dataset_group.add_argument(
        "--speaker_independent", action="store_true",
        help="Split berdasarkan speaker (speaker test tidak muncul di training)",
    )

    # --- Preprocessing ---
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Target sample rate audio (Hz)",
    )
    preproc_group.add_argument(
        "--max_audio_length", type=float, default=10.0,
        help="Durasi maksimum audio (detik)",
    )
    preproc_group.add_argument(
        "--n_mels", type=int, default=80,
        help="Jumlah Mel filter bank",
    )
    preproc_group.add_argument(
        "--feature_type", type=str, default="logmel", choices=["logmel", "mfcc"],
        help="Tipe fitur audio",
    )

    # --- Model ---
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--lstm_hidden", type=int, default=256,
        help="Ukuran hidden state BiLSTM",
    )
    model_group.add_argument(
        "--lstm_layers", type=int, default=3,
        help="Jumlah layer BiLSTM",
    )
    model_group.add_argument(
        "--lstm_dropout", type=float, default=0.3,
        help="Dropout antar layer BiLSTM",
    )
    model_group.add_argument(
        "--fc_dropout", type=float, default=0.3,
        help="Dropout sebelum layer proyeksi",
    )

    # --- Training ---
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--batch_size", type=int, default=8,
        help="Ukuran batch",
    )
    train_group.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Learning rate awal (Adam)",
    )
    train_group.add_argument(
        "--epochs", type=int, default=50,
        help="Jumlah epoch maksimum",
    )
    train_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed untuk reproducibility",
    )
    train_group.add_argument(
        "--early_stop_patience", type=int, default=10,
        help="Epoch tanpa improvement sebelum early stopping",
    )
    train_group.add_argument(
        "--lr_patience", type=int, default=5,
        help="Epoch stagnan sebelum LR diturunkan",
    )
    train_group.add_argument(
        "--lr_factor", type=float, default=0.5,
        help="Faktor pengurangan LR (new_lr = lr * factor)",
    )
    train_group.add_argument(
        "--grad_clip", type=float, default=5.0,
        help="Gradient clipping (0 = nonaktif)",
    )
    train_group.add_argument(
        "--num_workers", type=int, default=0,
        help="Jumlah worker DataLoader (0 = main process, aman di Windows)",
    )

    # --- Checkpoint & Resume ---
    ckpt_group = parser.add_argument_group("Checkpoint")
    ckpt_group.add_argument(
        "--checkpoint_dir", type=Path, default=Path("models"),
        help="Direktori penyimpanan model checkpoint",
    )
    ckpt_group.add_argument(
        "--results_dir", type=Path, default=Path("results"),
        help="Direktori penyimpanan log dan history",
    )
    ckpt_group.add_argument(
        "--resume", type=Path, default=None,
        help="Path checkpoint untuk melanjutkan training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Import setelah argparse (agar --help tidak memuat PyTorch)
    # ------------------------------------------------------------------
    import torch
    from src.preprocessing import AudioPreprocessor
    from src.dataset import DatasetManager
    from src.model import SpeechRecognitionModel, ModelConfig
    from src.trainer import Trainer, TrainingConfig

    logger.info("=" * 65)
    logger.info("SPEECH-TO-TEXT BAHASA BENGKULU — TRAINING")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device  : %s", device)
    if device.type == "cuda":
        logger.info("GPU     : %s", torch.cuda.get_device_name(device))
        logger.info("VRAM    : %.1f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)
    else:
        logger.info("GPU tidak tersedia — training menggunakan CPU (lebih lambat).")

    # ------------------------------------------------------------------
    # 2. Seed
    # ------------------------------------------------------------------
    Trainer.set_seed(args.seed)

    # ------------------------------------------------------------------
    # 3. Cek dataset
    # ------------------------------------------------------------------
    metadata_path = args.dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error(
            "metadata.csv tidak ditemukan di %s\n"
            "Langkah:\n"
            "  1. Tambahkan file audio ke dataset/audio/\n"
            "  2. Buat dataset/metadata.csv (kolom: file_name,text,speaker_id)\n"
            "  3. Jalankan: python prepare_dataset.py\n"
            "  4. Jalankan ulang: python train.py",
            args.dataset_dir,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Preprocessor
    # ------------------------------------------------------------------
    preprocessor = AudioPreprocessor(
        target_sample_rate=args.sample_rate,
        max_duration_sec=args.max_audio_length,
        n_mels=args.n_mels,
        feature_type=args.feature_type,
    )
    logger.info("Preprocessor: %s", preprocessor)

    # ------------------------------------------------------------------
    # 5. Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset_manager = DatasetManager(
        dataset_dir=args.dataset_dir,
        preprocessor=preprocessor,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        speaker_independent=args.speaker_independent,
        seed=args.seed,
    )

    try:
        train_loader, val_loader, test_loader = dataset_manager.get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        logger.error("Gagal menyiapkan DataLoader: %s", exc)
        sys.exit(1)

    # Simpan vocab agar bisa digunakan saat inference
    vocab_path = args.checkpoint_dir / "vocab.txt"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_manager.tokenizer.save(vocab_path)
    logger.info("Vocab disimpan ke %s (vocab_size=%d)", vocab_path, dataset_manager.tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model_config = ModelConfig(
        vocab_size=dataset_manager.tokenizer.vocab_size,
        n_mels=args.n_mels,
        cnn_channels=[32, 64, 128],
        cnn_kernel_size=(3, 3),
        cnn_stride=(1, 1),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        fc_dropout=args.fc_dropout,
    )

    model = SpeechRecognitionModel(model_config)
    logger.info("Model: %s", model)
    logger.info(
        "Parameter trainable: %s",
        f"{model.get_num_parameters():,}",
    )

    # ------------------------------------------------------------------
    # 7. Training config
    # ------------------------------------------------------------------
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        sample_rate=args.sample_rate,
        max_audio_length=args.max_audio_length,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        grad_clip=args.grad_clip,
        checkpoint_dir=str(args.checkpoint_dir),
        results_dir=str(args.results_dir),
    )

    # ------------------------------------------------------------------
    # 8. Trainer & fit
    # ------------------------------------------------------------------
    trainer = Trainer(model=model, config=training_config, device=device)
    logger.info("Device info: %s", trainer.get_device_info())

    logger.info("-" * 65)
    logger.info("Mulai training...")
    logger.info("  Batch size      : %d", args.batch_size)
    logger.info("  Learning rate   : %.1e", args.learning_rate)
    logger.info("  Epochs maks     : %d", args.epochs)
    logger.info("  Early stop      : %d epoch", args.early_stop_patience)
    logger.info("  Train batches   : %d", len(train_loader))
    logger.info("  Val batches     : %d", len(val_loader))
    logger.info("-" * 65)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )

    # ------------------------------------------------------------------
    # 9. Ringkasan akhir
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("TRAINING SELESAI")
    logger.info("=" * 65)
    logger.info("Best validation loss : %.4f", trainer.best_val_loss)
    logger.info("Total epoch dijalankan: %d", len(history["train_loss"]))
    logger.info("Model terbaik tersimpan di: %s", args.checkpoint_dir / "best_model.pt")
    logger.info("")
    logger.info("Langkah selanjutnya:")
    logger.info("  Evaluasi : python evaluate.py")
    logger.info("  Aplikasi : streamlit run app.py")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
