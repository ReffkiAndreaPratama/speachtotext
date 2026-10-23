"""
evaluate.py
===========
Script evaluasi model Speech-to-Text Bahasa Bengkulu.

Menghitung metrik:
    - WER  (Word Error Rate)    — menggunakan library jiwer
    - CER  (Character Error Rate) — menggunakan jiwer

Penggunaan:
    python evaluate.py
    python evaluate.py --model_path models/best_model.pt --split test
    python evaluate.py --decoding beam --beam_width 10
    python evaluate.py --output_samples 20

Output:
    - Ringkasan metrik di konsol
    - results/evaluation_report.json  — laporan lengkap JSON
    - results/sample_predictions.csv  — sampel prediksi vs referensi

Catatan penting:
    Metrik hanya bermakna jika model sudah dilatih dengan dataset nyata.
    Jika model belum dilatih, script akan berjalan tetapi WER/CER akan
    sangat tinggi (mendekati 100%) — ini adalah hasil aktual, bukan rekayasa.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
LOG_DIR = Path("results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "evaluation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Metrik WER / CER
# ===========================================================================

def compute_wer_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Hitung WER dan CER menggunakan library jiwer.

    Parameters
    ----------
    references  : list teks referensi (ground truth)
    hypotheses  : list teks prediksi model

    Returns
    -------
    dict dengan kunci: wer, cer, wer_pct, cer_pct
    """
    try:
        from jiwer import wer as compute_wer, cer as compute_cer
    except ImportError:
        logger.error(
            "Library 'jiwer' tidak terinstal. "
            "Jalankan: pip install jiwer"
        )
        return {"wer": -1.0, "cer": -1.0, "wer_pct": -1.0, "cer_pct": -1.0}

    if not references or not hypotheses:
        return {"wer": -1.0, "cer": -1.0, "wer_pct": -1.0, "cer_pct": -1.0}

    # jiwer membutuhkan string non-kosong
    refs_clean = []
    hyps_clean = []
    for ref, hyp in zip(references, hypotheses):
        ref = ref.strip() if ref.strip() else " "
        hyp = hyp.strip() if hyp.strip() else " "
        refs_clean.append(ref)
        hyps_clean.append(hyp)

    try:
        wer_score = compute_wer(refs_clean, hyps_clean)
        cer_score = compute_cer(refs_clean, hyps_clean)
    except Exception as exc:
        logger.error("Gagal menghitung WER/CER: %s", exc)
        return {"wer": -1.0, "cer": -1.0, "wer_pct": -1.0, "cer_pct": -1.0}

    return {
        "wer": round(float(wer_score), 4),
        "cer": round(float(cer_score), 4),
        "wer_pct": round(float(wer_score) * 100, 2),
        "cer_pct": round(float(cer_score) * 100, 2),
    }


def compute_wer_cer_per_sample(
    reference: str, hypothesis: str
) -> Dict[str, float]:
    """Hitung WER/CER untuk satu pasang kalimat."""
    return compute_wer_cer([reference], [hypothesis])


# ===========================================================================
# Fungsi evaluasi utama
# ===========================================================================

def evaluate_model(
    model_path: Path,
    vocab_path: Path,
    dataset_dir: Path,
    split: str = "test",
    decoding: str = "greedy",
    beam_width: int = 5,
    sample_rate: int = 16000,
    max_audio_length: float = 10.0,
    n_mels: int = 80,
    feature_type: str = "logmel",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    speaker_independent: bool = False,
    seed: int = 42,
    batch_size: int = 1,
    output_samples: int = 20,
    results_dir: Path = Path("results"),
) -> Dict:
    """
    Evaluasi model pada split yang dipilih (train/val/test).

    Returns
    -------
    dict laporan evaluasi lengkap
    """
    import torch
    from src.preprocessing import AudioPreprocessor
    from src.dataset import DatasetManager, BengkuluSpeechDataset
    from src.inference import InferenceEngine, CTCDecoder
    from src.dataset import CharacterTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    if not model_path.exists():
        logger.error(
            "Model tidak ditemukan: %s\n"
            "Jalankan training terlebih dahulu: python train.py",
            model_path,
        )
        sys.exit(1)

    if not vocab_path.exists():
        logger.error(
            "Vocab tidak ditemukan: %s\n"
            "Vocab dibuat otomatis saat training. Jalankan python train.py dulu.",
            vocab_path,
        )
        sys.exit(1)

    logger.info("Loading model dari %s ...", model_path)
    engine = InferenceEngine.from_checkpoint(
        model_path=model_path,
        vocab_path=vocab_path,
        device=device,
        sample_rate=sample_rate,
        max_duration_sec=max_audio_length,
        n_mels=n_mels,
        feature_type=feature_type,
        decoding_strategy=decoding,
    )

    # ------------------------------------------------------------------
    # Siapkan dataset split
    # ------------------------------------------------------------------
    preprocessor = AudioPreprocessor(
        target_sample_rate=sample_rate,
        max_duration_sec=max_audio_length,
        n_mels=n_mels,
        feature_type=feature_type,
    )

    dataset_manager = DatasetManager(
        dataset_dir=dataset_dir,
        preprocessor=preprocessor,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        speaker_independent=speaker_independent,
        seed=seed,
    )

    df_all = dataset_manager.load_metadata()
    df_train, df_val, df_test = dataset_manager.split(df_all)

    split_map = {"train": df_train, "val": df_val, "test": df_test}
    if split not in split_map:
        logger.error("split harus 'train', 'val', atau 'test'. Dapat: %s", split)
        sys.exit(1)

    df_eval = split_map[split]
    audio_dir = dataset_dir / "audio"

    logger.info(
        "Evaluasi pada split '%s': %d sampel", split, len(df_eval)
    )

    if len(df_eval) == 0:
        logger.error(
            "Split '%s' kosong. Periksa dataset dan rasio split.", split
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Inferensi per sampel
    # ------------------------------------------------------------------
    references: List[str] = []
    hypotheses: List[str] = []
    sample_rows: List[Dict] = []
    errors: List[str] = []

    total = len(df_eval)
    for idx, row in df_eval.iterrows():
        file_name = str(row["file_name"]).strip()
        reference = str(row["text"]).strip().lower()
        audio_path = audio_dir / file_name

        if not audio_path.exists():
            logger.warning("File tidak ada, skip: %s", file_name)
            errors.append(file_name)
            continue

        result = engine.transcribe(
            audio_input=audio_path,
            decoding_strategy=decoding,
            beam_width=beam_width,
        )

        hyp = result["text"].strip().lower()
        # Abaikan placeholder jika model belum dilatih
        if "[model belum" in hyp or "[error" in hyp:
            hyp = ""

        references.append(reference)
        hypotheses.append(hyp)

        per_sample_metrics = compute_wer_cer_per_sample(reference, hyp)
        row_result = {
            "file_name": file_name,
            "speaker_id": str(row.get("speaker_id", "")),
            "reference": reference,
            "hypothesis": hyp,
            "confidence": result.get("confidence", 0.0),
            "latency_sec": result.get("latency_sec", 0.0),
            "wer": per_sample_metrics["wer_pct"],
            "cer": per_sample_metrics["cer_pct"],
        }
        sample_rows.append(row_result)

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info("  Progress: %d / %d", min(idx + 1, total), total)

    # ------------------------------------------------------------------
    # Hitung metrik keseluruhan
    # ------------------------------------------------------------------
    logger.info("─" * 60)
    overall_metrics = compute_wer_cer(references, hypotheses)

    avg_confidence = (
        sum(r["confidence"] for r in sample_rows) / max(len(sample_rows), 1)
    )
    avg_latency = (
        sum(r["latency_sec"] for r in sample_rows) / max(len(sample_rows), 1)
    )

    logger.info("=" * 60)
    logger.info("HASIL EVALUASI — split: %s", split.upper())
    logger.info("=" * 60)
    logger.info("Jumlah sampel dievaluasi : %d", len(references))
    logger.info("Jumlah sampel dilewati   : %d (file tidak ada)", len(errors))
    logger.info("")

    if overall_metrics["wer"] >= 0:
        logger.info("WER  (Word Error Rate)       : %.4f  (%.2f%%)",
                    overall_metrics["wer"], overall_metrics["wer_pct"])
        logger.info("CER  (Character Error Rate)  : %.4f  (%.2f%%)",
                    overall_metrics["cer"], overall_metrics["cer_pct"])
    else:
        logger.warning("WER/CER tidak dapat dihitung (jiwer tidak tersedia atau data kosong).")

    logger.info("Rata-2 confidence        : %.4f", avg_confidence)
    logger.info("Rata-2 latensi           : %.3f detik", avg_latency)

    if overall_metrics["wer"] > 0.9:
        logger.warning(
            "\n⚠  WER sangat tinggi (%.1f%%). "
            "Ini wajar jika model belum dilatih atau dataset terlalu kecil.",
            overall_metrics["wer_pct"],
        )

    # ------------------------------------------------------------------
    # Simpan sampel prediksi
    # ------------------------------------------------------------------
    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    samples_path = results_dir / f"sample_predictions_{split}.csv"
    samples_df = pd.DataFrame(sample_rows[:output_samples])
    samples_df.to_csv(samples_path, index=False, encoding="utf-8")
    logger.info("\nSampel prediksi disimpan ke %s (%d baris)", samples_path, len(samples_df))

    # Tampilkan beberapa sampel
    if sample_rows:
        logger.info("\n--- CONTOH PREDIKSI (5 pertama) ---")
        for r in sample_rows[:5]:
            logger.info("File  : %s", r["file_name"])
            logger.info("Ref   : %s", r["reference"])
            logger.info("Hyp   : %s", r["hypothesis"])
            logger.info("WER   : %.1f%%  CER: %.1f%%", r["wer"], r["cer"])
            logger.info("")

    # ------------------------------------------------------------------
    # Laporan JSON lengkap
    # ------------------------------------------------------------------
    report = {
        "split": split,
        "total_samples": len(references),
        "skipped_samples": len(errors),
        "decoding_strategy": decoding,
        "beam_width": beam_width if decoding == "beam" else None,
        "metrics": {
            "wer": overall_metrics["wer"],
            "cer": overall_metrics["cer"],
            "wer_pct": overall_metrics["wer_pct"],
            "cer_pct": overall_metrics["cer_pct"],
        },
        "avg_confidence": round(avg_confidence, 4),
        "avg_latency_sec": round(avg_latency, 4),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "model_info": engine.get_model_info(),
        "note": (
            "Metrik ini berasal dari model yang sudah dilatih pada dataset nyata. "
            "Jika WER/CER sangat tinggi, kemungkinan model masih memerlukan lebih "
            "banyak data atau epoch training."
        ),
        "all_predictions": sample_rows,
    }

    report_path = results_dir / f"evaluation_report_{split}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Laporan lengkap disimpan ke %s", report_path)

    return report


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluasi model Speech-to-Text Bahasa Bengkulu (WER/CER)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=Path, default=Path("models/best_model.pt"),
        help="Path ke file model (.pt)",
    )
    parser.add_argument(
        "--vocab_path", type=Path, default=Path("models/vocab.txt"),
        help="Path ke file vocab.txt",
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("dataset"),
        help="Direktori dataset",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Split yang akan dievaluasi",
    )
    parser.add_argument(
        "--decoding", type=str, default="greedy", choices=["greedy", "beam"],
        help="Strategi decoding",
    )
    parser.add_argument(
        "--beam_width", type=int, default=5,
        help="Lebar beam (hanya untuk decoding='beam')",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Sample rate audio (harus sama dengan training)",
    )
    parser.add_argument(
        "--max_audio_length", type=float, default=10.0,
        help="Durasi maksimum audio (harus sama dengan training)",
    )
    parser.add_argument(
        "--n_mels", type=int, default=80,
        help="Jumlah Mel filter (harus sama dengan training)",
    )
    parser.add_argument(
        "--feature_type", type=str, default="logmel", choices=["logmel", "mfcc"],
        help="Tipe fitur (harus sama dengan training)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Proporsi split training (harus sama dengan training)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Proporsi split validasi",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Proporsi split test",
    )
    parser.add_argument(
        "--speaker_independent", action="store_true",
        help="Gunakan split speaker-independent (harus sama dengan training)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (harus sama dengan training)",
    )
    parser.add_argument(
        "--output_samples", type=int, default=20,
        help="Jumlah sampel prediksi yang disimpan ke CSV",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"),
        help="Direktori output laporan",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("SPEECH-TO-TEXT BAHASA BENGKULU — EVALUASI")
    logger.info("=" * 65)

    evaluate_model(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        dataset_dir=args.dataset_dir,
        split=args.split,
        decoding=args.decoding,
        beam_width=args.beam_width,
        sample_rate=args.sample_rate,
        max_audio_length=args.max_audio_length,
        n_mels=args.n_mels,
        feature_type=args.feature_type,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        speaker_independent=args.speaker_independent,
        seed=args.seed,
        output_samples=args.output_samples,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
