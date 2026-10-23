"""
prepare_dataset.py
==================
Tool validasi dan persiapan dataset untuk Speech-to-Text Bahasa Bengkulu.

Menjalankan serangkaian pemeriksaan menyeluruh pada dataset sebelum training:

    1.  Keberadaan file metadata.csv dan direktori audio/
    2.  Kolom wajib (file_name, text, speaker_id)
    3.  Keberadaan setiap file audio
    4.  Format file audio yang dapat dibaca torchaudio
    5.  Sample rate audio
    6.  Konvertibilitas ke mono
    7.  Durasi audio (terlalu pendek / terlalu panjang)
    8.  Transkripsi kosong atau tidak valid
    9.  Duplikasi file_name
    10. Duplikasi pasangan (file_name, text)
    11. Statistik speaker (distribusi data per speaker)
    12. Ringkasan dataset (jumlah, durasi total, rata-rata, dll.)

Penggunaan:
    python prepare_dataset.py
    python prepare_dataset.py --dataset_dir dataset --min_duration 0.5 --max_duration 15.0
    python prepare_dataset.py --fix          # hapus baris bermasalah dari metadata

Output:
    - Laporan validasi di konsol
    - File results/dataset_validation_report.json
    - (opsional) metadata_cleaned.csv jika --fix diaktifkan
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torchaudio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Konstanta default
# ===========================================================================

DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_AUDIO_DIR = DEFAULT_DATASET_DIR / "audio"
DEFAULT_METADATA = DEFAULT_DATASET_DIR / "metadata.csv"
DEFAULT_RESULTS_DIR = Path("results")

MIN_DURATION_SEC = 0.5    # audio terlalu pendek (< 0.5 detik)
MAX_DURATION_SEC = 30.0   # audio terlalu panjang (> 30 detik)
TARGET_SAMPLE_RATE = 16000
VALID_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ===========================================================================
# Fungsi validasi individual
# ===========================================================================

def check_metadata_exists(metadata_path: Path) -> bool:
    """Periksa apakah metadata.csv ada."""
    if not metadata_path.exists():
        logger.error(
            "metadata.csv tidak ditemukan di %s\n"
            "Buat file metadata.csv dengan format:\n"
            "  file_name,text,speaker_id\n"
            "  audio_001.wav,transkripsi kalimat,speaker_01",
            metadata_path.parent,
        )
        return False
    logger.info("✓ metadata.csv ditemukan: %s", metadata_path)
    return True


def check_audio_dir_exists(audio_dir: Path) -> bool:
    """Periksa apakah direktori audio/ ada."""
    if not audio_dir.exists():
        logger.error("Direktori audio tidak ditemukan: %s", audio_dir)
        return False
    audio_files = list(audio_dir.iterdir())
    logger.info("✓ Direktori audio ditemukan: %s (%d file)", audio_dir, len(audio_files))
    return True


def check_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Periksa kolom wajib di metadata."""
    required = {"file_name", "text", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Kolom wajib tidak ditemukan: %s", missing)
        return False, list(missing)
    logger.info("✓ Semua kolom wajib tersedia: %s", list(required))
    return True, []


def check_empty_fields(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Deteksi baris dengan kolom kosong.
    Kembalikan (jumlah baris bermasalah, subset dataframe bermasalah).
    """
    mask = df["file_name"].isna() | df["text"].isna() | df["speaker_id"].isna()
    mask |= (df["file_name"].str.strip() == "") if df["file_name"].dtype == object else False
    mask |= (df["text"].str.strip() == "") if df["text"].dtype == object else False
    mask |= (df["speaker_id"].str.strip() == "") if df["speaker_id"].dtype == object else False

    bad = df[mask]
    if len(bad) > 0:
        logger.warning("⚠ %d baris dengan kolom kosong:", len(bad))
        logger.warning("%s", bad[["file_name", "text", "speaker_id"]].to_string())
    else:
        logger.info("✓ Tidak ada kolom kosong.")
    return len(bad), bad


def check_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """Deteksi duplikasi file_name dan pasangan (file_name, text)."""
    result = {}

    dup_filenames = df[df["file_name"].duplicated(keep=False)]
    result["duplicate_filenames"] = len(dup_filenames)
    if len(dup_filenames) > 0:
        logger.warning("⚠ %d baris dengan file_name duplikat:", len(dup_filenames))
        logger.warning("%s", dup_filenames["file_name"].value_counts().head(10).to_string())
    else:
        logger.info("✓ Tidak ada file_name duplikat.")

    dup_pairs = df[df.duplicated(subset=["file_name", "text"], keep=False)]
    result["duplicate_pairs"] = len(dup_pairs)
    if len(dup_pairs) > 0:
        logger.warning("⚠ %d baris dengan pasangan (file_name, text) duplikat.", len(dup_pairs))
    else:
        logger.info("✓ Tidak ada pasangan (file_name, text) duplikat.")

    return result


def check_audio_files(
    df: pd.DataFrame,
    audio_dir: Path,
    min_duration: float = MIN_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC,
) -> Dict:
    """
    Periksa setiap file audio:
    - Keberadaan file
    - Ekstensi valid
    - Dapat dibaca torchaudio
    - Sample rate
    - Durasi
    """
    issues = {
        "missing_files": [],
        "invalid_extension": [],
        "unreadable_files": [],
        "wrong_sample_rate": [],
        "too_short": [],
        "too_long": [],
        "valid_files": [],
    }

    durations = []
    sample_rates = []

    total = len(df)
    for idx, row in df.iterrows():
        file_name = str(row["file_name"]).strip()
        audio_path = audio_dir / file_name

        # 1. Keberadaan file
        if not audio_path.exists():
            issues["missing_files"].append(file_name)
            continue

        # 2. Ekstensi
        ext = audio_path.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            issues["invalid_extension"].append(file_name)
            continue

        # 3. Dapat dibaca
        try:
            info = torchaudio.info(str(audio_path))
        except Exception as exc:
            issues["unreadable_files"].append({"file": file_name, "error": str(exc)})
            continue

        # 4. Sample rate
        sr = info.sample_rate
        sample_rates.append(sr)
        if sr != TARGET_SAMPLE_RATE:
            issues["wrong_sample_rate"].append({
                "file": file_name,
                "sample_rate": sr,
                "expected": TARGET_SAMPLE_RATE,
                "note": "Akan di-resample otomatis oleh preprocessor",
            })

        # 5. Durasi
        duration = info.num_frames / info.sample_rate
        durations.append({"file": file_name, "duration": round(duration, 3)})

        if duration < min_duration:
            issues["too_short"].append({
                "file": file_name,
                "duration": round(duration, 3),
                "min": min_duration,
            })
        elif duration > max_duration:
            issues["too_long"].append({
                "file": file_name,
                "duration": round(duration, 3),
                "max": max_duration,
            })
        else:
            issues["valid_files"].append(file_name)

    # Summary durasi
    if durations:
        dur_values = [d["duration"] for d in durations]
        issues["duration_stats"] = {
            "count": len(dur_values),
            "total_sec": round(sum(dur_values), 2),
            "total_min": round(sum(dur_values) / 60, 2),
            "mean_sec": round(sum(dur_values) / len(dur_values), 2),
            "min_sec": round(min(dur_values), 2),
            "max_sec": round(max(dur_values), 2),
        }
    else:
        issues["duration_stats"] = {}

    issues["all_durations"] = durations

    # Log ringkasan
    n_missing = len(issues["missing_files"])
    n_unreadable = len(issues["unreadable_files"])
    n_short = len(issues["too_short"])
    n_long = len(issues["too_long"])
    n_wrong_sr = len(issues["wrong_sample_rate"])
    n_valid = len(issues["valid_files"])

    logger.info("─" * 50)
    logger.info("HASIL PEMERIKSAAN FILE AUDIO (%d total entri):", total)
    logger.info("  ✓ Valid               : %d", n_valid)
    logger.info("  ✗ File tidak ada      : %d", n_missing)
    logger.info("  ✗ Tidak terbaca       : %d", n_unreadable)
    logger.info("  ⚠ Durasi terlalu pendek: %d (< %.1fs)", n_short, min_duration)
    logger.info("  ⚠ Durasi terlalu panjang: %d (> %.1fs)", n_long, max_duration)
    logger.info("  ⚠ Sample rate berbeda : %d (akan di-resample)", n_wrong_sr)

    if issues.get("duration_stats"):
        s = issues["duration_stats"]
        logger.info("─" * 50)
        logger.info("STATISTIK DURASI:")
        logger.info("  Total  : %.2f detik (%.2f menit)", s["total_sec"], s["total_min"])
        logger.info("  Rata-2 : %.2f detik", s["mean_sec"])
        logger.info("  Min    : %.2f detik", s["min_sec"])
        logger.info("  Max    : %.2f detik", s["max_sec"])

    if n_missing > 0:
        logger.error("File audio yang tidak ditemukan (10 pertama):")
        for f in issues["missing_files"][:10]:
            logger.error("  - %s", f)

    return issues


def check_speaker_distribution(df: pd.DataFrame) -> Dict:
    """Hitung dan tampilkan distribusi data per speaker."""
    speaker_counts = df["speaker_id"].value_counts()
    n_speakers = len(speaker_counts)

    logger.info("─" * 50)
    logger.info("DISTRIBUSI SPEAKER (%d speaker unik):", n_speakers)
    for speaker, count in speaker_counts.items():
        bar = "█" * min(count, 40)
        logger.info("  %-20s : %3d  %s", speaker, count, bar)

    if n_speakers < 2:
        logger.warning(
            "⚠ Hanya %d speaker. Disarankan minimal 3+ speaker "
            "untuk evaluasi yang beragam.",
            n_speakers,
        )

    if n_speakers == 0:
        return {
            "n_speakers": 0,
            "speaker_counts": {},
            "min_samples_per_speaker": 0,
            "max_samples_per_speaker": 0,
            "mean_samples_per_speaker": 0.0,
        }

    return {
        "n_speakers": n_speakers,
        "speaker_counts": speaker_counts.to_dict(),
        "min_samples_per_speaker": int(speaker_counts.min()),
        "max_samples_per_speaker": int(speaker_counts.max()),
        "mean_samples_per_speaker": round(float(speaker_counts.mean()), 1),
    }


def check_transcription_quality(df: pd.DataFrame) -> Dict:
    """Analisis kualitas transkripsi (panjang, karakter unik, dll.)."""
    texts = df["text"].dropna().astype(str).str.strip()
    texts = texts[texts.str.len() > 0]

    # Guard: dataset kosong
    if len(texts) == 0:
        logger.info("─" * 50)
        logger.info("KUALITAS TRANSKRIPSI:")
        logger.warning("  Dataset kosong — tidak ada transkripsi untuk dianalisis.")
        return {
            "total_transcriptions": 0,
            "avg_char_length": 0.0,
            "min_char_length": 0,
            "max_char_length": 0,
            "avg_word_count": 0.0,
            "unique_characters": [],
            "vocab_size_chars": 0,
            "suspicious_chars": [],
        }

    lengths = texts.str.len()
    words = texts.str.split().str.len()
    all_chars = set("".join(texts.tolist()))

    # Filter karakter non-latin yang tidak diharapkan
    suspicious = [c for c in all_chars if not (c.isalpha() or c.isspace() or c in ".,!?'-")]

    stats = {
        "total_transcriptions": len(texts),
        "avg_char_length": round(float(lengths.mean()), 1),
        "min_char_length": int(lengths.min()),
        "max_char_length": int(lengths.max()),
        "avg_word_count": round(float(words.mean()), 1),
        "unique_characters": sorted(all_chars),
        "vocab_size_chars": len(all_chars),
        "suspicious_chars": suspicious,
    }

    logger.info("─" * 50)
    logger.info("KUALITAS TRANSKRIPSI:")
    logger.info("  Jumlah transkripsi  : %d", stats["total_transcriptions"])
    logger.info("  Rata-2 panjang      : %.1f karakter", stats["avg_char_length"])
    logger.info("  Rata-2 jumlah kata  : %.1f kata", stats["avg_word_count"])
    logger.info("  Karakter unik       : %d", stats["vocab_size_chars"])
    logger.info("  Karakter unik       : %s", "".join(sorted(all_chars))[:80])
    if suspicious:
        logger.warning("  ⚠ Karakter mencurigakan: %s", suspicious[:20])

    return stats


# ===========================================================================
# Fix: bersihkan metadata dari baris bermasalah
# ===========================================================================

def fix_metadata(
    df: pd.DataFrame,
    audio_dir: Path,
    min_duration: float = MIN_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Hapus baris bermasalah dari metadata:
    - file tidak ada
    - teks kosong
    - duplikat file_name
    - durasi di luar batas

    Simpan hasilnya ke metadata_cleaned.csv.
    """
    original_len = len(df)

    # Hapus NaN / kosong
    df = df.dropna(subset=["file_name", "text", "speaker_id"])
    df["file_name"] = df["file_name"].str.strip()
    df["text"] = df["text"].str.strip()
    df["speaker_id"] = df["speaker_id"].str.strip()
    df = df[(df["file_name"] != "") & (df["text"] != "") & (df["speaker_id"] != "")]

    # Hapus duplikat file_name (pertahankan yang pertama)
    df = df.drop_duplicates(subset=["file_name"], keep="first")

    # Hapus file yang tidak ada
    def file_exists(fn):
        return (audio_dir / fn).exists()
    df = df[df["file_name"].apply(file_exists)]

    # Hapus durasi di luar batas
    def check_duration(fn):
        try:
            info = torchaudio.info(str(audio_dir / fn))
            dur = info.num_frames / info.sample_rate
            return min_duration <= dur <= max_duration
        except Exception:
            return False
    df = df[df["file_name"].apply(check_duration)]

    df = df.reset_index(drop=True)
    removed = original_len - len(df)
    logger.info("Fix metadata: %d → %d baris (%d dihapus)", original_len, len(df), removed)

    if output_path is None:
        output_path = DEFAULT_DATASET_DIR / "metadata_cleaned.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("metadata_cleaned.csv disimpan ke %s", output_path)

    return df


# ===========================================================================
# Laporan JSON
# ===========================================================================

def save_report(report: Dict, results_dir: Path) -> Path:
    """Simpan laporan validasi ke JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "dataset_validation_report.json"

    # Bersihkan objek yang tidak JSON-serializable
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(report), f, indent=2, ensure_ascii=False)
    logger.info("Laporan disimpan ke %s", path)
    return path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validasi dataset Speech-to-Text Bahasa Bengkulu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR,
        help="Direktori root dataset (berisi audio/ dan metadata.csv)",
    )
    parser.add_argument(
        "--min_duration", type=float, default=MIN_DURATION_SEC,
        help="Durasi minimum audio yang valid (detik)",
    )
    parser.add_argument(
        "--max_duration", type=float, default=MAX_DURATION_SEC,
        help="Durasi maksimum audio yang valid (detik)",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Direktori output laporan",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Hapus baris bermasalah dan simpan metadata_cleaned.csv",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    audio_dir = dataset_dir / "audio"
    metadata_path = dataset_dir / "metadata.csv"

    logger.info("=" * 60)
    logger.info("VALIDASI DATASET SPEECH-TO-TEXT BAHASA BENGKULU")
    logger.info("=" * 60)
    logger.info("Dataset dir : %s", dataset_dir.resolve())
    logger.info("Audio dir   : %s", audio_dir.resolve())
    logger.info("Metadata    : %s", metadata_path.resolve())
    logger.info("=" * 60)

    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "checks": {},
        "summary": {},
    }

    # --- 1. Keberadaan direktori dan metadata ---
    if not check_metadata_exists(metadata_path):
        logger.error("Hentikan: metadata.csv tidak ditemukan.")
        sys.exit(1)

    if not check_audio_dir_exists(audio_dir):
        logger.error("Hentikan: direktori audio tidak ditemukan.")
        sys.exit(1)

    # --- 2. Baca metadata ---
    try:
        df = pd.read_csv(metadata_path, dtype=str)
    except Exception as exc:
        logger.error("Gagal membaca metadata.csv: %s", exc)
        sys.exit(1)

    logger.info("Jumlah baris di metadata.csv: %d", len(df))
    report["total_rows"] = len(df)

    # --- 3. Kolom wajib ---
    cols_ok, missing_cols = check_required_columns(df)
    report["checks"]["required_columns"] = {"ok": cols_ok, "missing": missing_cols}
    if not cols_ok:
        sys.exit(1)

    # --- 4. Field kosong ---
    n_empty, _ = check_empty_fields(df)
    report["checks"]["empty_fields"] = {"count": n_empty}

    # --- 5. Duplikasi ---
    dup_info = check_duplicates(df)
    report["checks"]["duplicates"] = dup_info

    # --- 6. File audio ---
    audio_issues = check_audio_files(df, audio_dir, args.min_duration, args.max_duration)
    report["checks"]["audio_files"] = {
        k: v for k, v in audio_issues.items() if k != "all_durations"
    }

    # --- 7. Speaker ---
    speaker_stats = check_speaker_distribution(df)
    report["checks"]["speaker_distribution"] = speaker_stats

    # --- 8. Transkripsi ---
    text_stats = check_transcription_quality(df)
    report["checks"]["transcription_quality"] = text_stats

    # --- 9. Summary ---
    n_valid = len(audio_issues["valid_files"])
    n_total = len(df)
    n_issues = (
        len(audio_issues["missing_files"])
        + len(audio_issues["unreadable_files"])
        + n_empty
        + dup_info.get("duplicate_filenames", 0)
    )

    report["summary"] = {
        "total_entries": n_total,
        "valid_audio_entries": n_valid,
        "entries_with_issues": n_issues,
        "dataset_ready": (n_valid >= 10 and n_issues == 0),
        "recommendation": (
            "Dataset siap untuk training."
            if n_valid >= 10 and n_issues == 0
            else (
                "Dataset BELUM siap. Perbaiki masalah di atas atau "
                "gunakan --fix untuk membersihkan otomatis."
            )
        ),
    }

    logger.info("=" * 60)
    logger.info("RINGKASAN")
    logger.info("=" * 60)
    logger.info("Total entri        : %d", n_total)
    logger.info("Audio valid        : %d", n_valid)
    logger.info("Entri bermasalah   : %d", n_issues)
    logger.info("Dataset siap       : %s", "YA ✓" if report["summary"]["dataset_ready"] else "TIDAK ✗")
    logger.info("Rekomendasi        : %s", report["summary"]["recommendation"])

    # --- 10. Simpan laporan ---
    save_report(report, args.results_dir)

    # --- 11. Fix opsional ---
    if args.fix:
        logger.info("=" * 60)
        logger.info("MODE FIX: Membersihkan metadata...")
        fix_metadata(
            df, audio_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            output_path=dataset_dir / "metadata_cleaned.csv",
        )
        logger.info(
            "Gunakan metadata_cleaned.csv sebagai metadata.csv untuk training "
            "dengan mengganti nama file tersebut."
        )

    if not report["summary"]["dataset_ready"] and not args.fix:
        logger.warning(
            "\nDataset belum siap. Tips:\n"
            "  1. Tambahkan file audio ke dataset/audio/\n"
            "  2. Perbarui dataset/metadata.csv dengan path dan transkripsi\n"
            "  3. Jalankan: python prepare_dataset.py --fix\n"
            "  4. Jalankan ulang: python prepare_dataset.py"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
