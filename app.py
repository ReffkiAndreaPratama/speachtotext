"""
app.py
======
Aplikasi web Streamlit untuk Speech-to-Text Bahasa Bengkulu.

Jalankan:
    streamlit run app.py

Halaman:
    1. 🎙️ Speech Recognition  — transkripsi audio
    2. 📊 Dataset Overview    — statistik dataset
    3. 🏋️ Training            — jalankan training dari UI
    4. 📈 Evaluation          — tampilkan metrik WER/CER
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ─── Konfigurasi halaman ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech-to-Text Bahasa Bengkulu",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Path default ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
VOCAB_PATH = ROOT / "models" / "vocab.txt"
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def model_is_trained() -> bool:
    return MODEL_PATH.exists() and VOCAB_PATH.exists()


@st.cache_resource(show_spinner="Memuat model…")
def load_engine(model_path: str, vocab_path: str):
    """Load InferenceEngine — cache agar tidak reload setiap interaksi."""
    try:
        from src.inference import InferenceEngine
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = InferenceEngine.from_checkpoint(
            model_path=Path(model_path),
            vocab_path=Path(vocab_path),
            device=device,
        )
        return engine, None
    except Exception as exc:
        return None, str(exc)


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.1f}s" if m > 0 else f"{s:.1f}s"


# ─── CSS ringan ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #4c72d6;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.85rem; color: #555; }
    .result-box {
        background: #f8fff8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.1rem;
        min-height: 60px;
    }
    .warning-box {
        background: #fff8e1;
        border: 1px solid #ffa000;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg",
             caption="Provinsi Bengkulu", use_column_width=True)

    st.markdown("## 🎙️ Speech-to-Text\n### Bahasa Bengkulu")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=[
            "🎙️ Speech Recognition",
            "📊 Dataset Overview",
            "🏋️ Training",
            "📈 Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # Status model
    if model_is_trained():
        st.success("✅ Model tersedia")
    else:
        st.warning("⚠️ Model belum dilatih")

    # Status dataset
    meta_path = DATASET_DIR / "metadata.csv"
    if meta_path.exists():
        try:
            df_meta = pd.read_csv(meta_path, dtype=str)
            st.info(f"📁 Dataset: {len(df_meta)} sampel")
        except Exception:
            st.info("📁 Dataset: metadata ada")
    else:
        st.error("❌ Dataset belum disiapkan")

    st.markdown("---")
    st.caption("Informatics Practitioner Lecture Series\n\nTema: Bridging Global Infrastructure to Address Local Challenge with HPC\n\nAsst. Prof. Dr. Worawan Diaz Carballo\nThammasat University")


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1: SPEECH RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🎙️ Speech Recognition":

    st.markdown("# 🎙️ Speech-to-Text Bahasa Bengkulu")
    st.markdown(
        "Upload file audio berbahasa Bengkulu, lalu klik **TRANSCRIBE** "
        "untuk mengubahnya menjadi teks."
    )
    st.markdown("---")

    # ── Pengaturan decoding ──────────────────────────────────────────
    with st.expander("⚙️ Pengaturan Decoding", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            decoding = st.selectbox(
                "Strategi Decoding",
                ["greedy", "beam"],
                help="Greedy: cepat. Beam: lebih akurat (butuh waktu lebih).",
            )
        with col_s2:
            beam_width = st.slider(
                "Beam Width",
                min_value=2, max_value=20, value=5,
                disabled=(decoding == "greedy"),
                help="Hanya aktif untuk Beam Search.",
            )
        trim_silence = st.checkbox("Trim Silence", value=True,
                                   help="Potong keheningan di awal/akhir audio.")

    st.markdown("---")

    # ── Upload audio ─────────────────────────────────────────────────
    st.markdown("#### 📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Pilih file audio (WAV, FLAC, MP3)",
        type=["wav", "flac", "mp3", "ogg"],
        help="Format yang didukung: WAV (disarankan), FLAC, MP3, OGG",
    )

    if uploaded_file is not None:
        st.markdown("#### 🔊 Audio")
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        file_info_col1, file_info_col2 = st.columns(2)
        with file_info_col1:
            st.metric("Nama file", uploaded_file.name)
        with file_info_col2:
            size_kb = uploaded_file.size / 1024
            st.metric("Ukuran", f"{size_kb:.1f} KB")

        st.markdown("---")

        # ── Tombol TRANSCRIBE ────────────────────────────────────────
        transcribe_btn = st.button(
            "🔍 TRANSCRIBE",
            type="primary",
            use_container_width=True,
        )

        if transcribe_btn:
            if not model_is_trained():
                st.markdown("""
<div class="warning-box">
⚠️ <strong>Model belum dilatih.</strong><br>
Pipeline siap — tetapi memerlukan dataset Bahasa Bengkulu untuk menghasilkan transkripsi bermakna.<br><br>
Langkah berikutnya:<br>
1. Siapkan dataset audio Bahasa Bengkulu<br>
2. Buat <code>dataset/metadata.csv</code><br>
3. Jalankan: <code>python prepare_dataset.py</code><br>
4. Jalankan: <code>python train.py</code>
</div>
""", unsafe_allow_html=True)
            else:
                engine, err = load_engine(str(MODEL_PATH), str(VOCAB_PATH))
                if err:
                    st.error(f"Gagal memuat model: {err}")
                else:
                    with st.spinner("Memproses audio…"):
                        audio_bytes = uploaded_file.read()
                        suffix = Path(uploaded_file.name).suffix

                        result = engine.transcribe(
                            audio_input=audio_bytes,
                            audio_suffix=suffix,
                            decoding_strategy=decoding,
                            beam_width=beam_width,
                            trim_silence=trim_silence,
                        )

                    st.markdown("#### 📝 Hasil Transkripsi")
                    transcription = result["text"]
                    st.markdown(
                        f'<div class="result-box">{transcription}</div>',
                        unsafe_allow_html=True,
                    )

                    # Metadata
                    st.markdown("#### 📊 Detail")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Durasi Audio", format_duration(result["duration_sec"]))
                    m2.metric("Latensi Inferensi", f"{result['latency_sec']:.3f} s")
                    m3.metric("Confidence", f"{result['confidence']:.3f}")
                    m4.metric("Decoding", result["decoding"].capitalize())

                    if not result.get("model_ready", True):
                        st.warning(
                            "Confidence sangat rendah — model mungkin belum cukup dilatih "
                            "atau audio tidak sesuai bahasa training."
                        )

                    # Download TXT
                    st.markdown("---")
                    st.download_button(
                        label="⬇️ Download Hasil Transkripsi (.txt)",
                        data=transcription,
                        file_name=f"transkripsi_{Path(uploaded_file.name).stem}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

    else:
        st.markdown("""
<div class="warning-box">
📂 Belum ada audio yang diupload.<br>
Pilih file audio WAV/FLAC/MP3 menggunakan tombol di atas.
</div>
""", unsafe_allow_html=True)

        if not model_is_trained():
            st.info(
                "💡 Model belum dilatih. Halaman ini siap digunakan setelah training "
                "selesai dengan dataset Bahasa Bengkulu."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2: DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dataset Overview":

    st.markdown("# 📊 Dataset Overview")
    st.markdown("Statistik dan distribusi dataset Speech Bahasa Bengkulu.")
    st.markdown("---")

    meta_path = DATASET_DIR / "metadata.csv"

    if not meta_path.exists():
        st.error("metadata.csv tidak ditemukan.")
        st.markdown("""
**Cara menyiapkan dataset:**

```
dataset/
├── audio/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

Format `metadata.csv`:
```csv
file_name,text,speaker_id
audio_001.wav,contoh transkripsi,speaker_01
```

Setelah mengisi dataset, jalankan:
```bash
python prepare_dataset.py
```
""")
        st.stop()

    try:
        df = pd.read_csv(meta_path, dtype=str)
    except Exception as exc:
        st.error(f"Gagal membaca metadata.csv: {exc}")
        st.stop()

    # ── Metrik utama ─────────────────────────────────────────────────
    audio_dir = DATASET_DIR / "audio"
    n_samples = len(df)
    n_speakers = df["speaker_id"].nunique() if "speaker_id" in df.columns else 0

    # Hitung durasi
    durations = []
    for _, row in df.iterrows():
        if "file_name" not in row:
            continue
        p = audio_dir / str(row["file_name"]).strip()
        if p.exists():
            try:
                import torchaudio
                info = torchaudio.info(str(p))
                durations.append(info.num_frames / info.sample_rate)
            except Exception:
                pass

    total_dur = sum(durations)
    avg_dur = total_dur / max(len(durations), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sampel", f"{n_samples:,}")
    c2.metric("Jumlah Speaker", f"{n_speakers}")
    c3.metric("Total Durasi", format_duration(total_dur))
    c4.metric("Rata-2 Durasi", f"{avg_dur:.1f} s")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📁 Distribusi Speaker")
        if "speaker_id" in df.columns:
            speaker_counts = df["speaker_id"].value_counts().reset_index()
            speaker_counts.columns = ["Speaker", "Jumlah Sampel"]
            st.bar_chart(speaker_counts.set_index("Speaker"))
            st.dataframe(speaker_counts, use_container_width=True)

    with col_right:
        st.markdown("#### ⏱️ Distribusi Durasi Audio")
        if durations:
            dur_df = pd.DataFrame({"Durasi (detik)": durations})
            st.bar_chart(dur_df["Durasi (detik)"].value_counts(bins=10).sort_index())
            st.markdown(f"""
| Statistik | Nilai |
|-----------|-------|
| Min | {min(durations):.2f} s |
| Max | {max(durations):.2f} s |
| Rata-2 | {avg_dur:.2f} s |
| Total | {format_duration(total_dur)} |
""")
        else:
            st.info("File audio belum tersedia untuk menghitung durasi.")

    st.markdown("---")
    st.markdown("#### 📋 Tabel Metadata")
    st.dataframe(df.head(50), use_container_width=True)

    # Laporan validasi jika ada
    val_report = RESULTS_DIR / "dataset_validation_report.json"
    if val_report.exists():
        st.markdown("---")
        st.markdown("#### ✅ Laporan Validasi Terakhir")
        with open(val_report, encoding="utf-8") as f:
            report = json.load(f)
        summary = report.get("summary", {})
        ready = summary.get("dataset_ready", False)
        if ready:
            st.success(f"Dataset siap: {summary.get('valid_audio_entries', 0)} audio valid.")
        else:
            st.warning(summary.get("recommendation", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3: TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏋️ Training":

    st.markdown("# 🏋️ Training")
    st.markdown(
        "Konfigurasi dan jalankan training model dari antarmuka web. "
        "Atau gunakan CLI: `python train.py`"
    )
    st.markdown("---")

    if not (DATASET_DIR / "metadata.csv").exists():
        st.error("Dataset belum disiapkan. Siapkan dataset terlebih dahulu di halaman **Dataset Overview**.")
        st.stop()

    # ── Form parameter ───────────────────────────────────────────────
    st.markdown("#### ⚙️ Hyperparameter Training")

    with st.form("training_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=8)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=0.1,
                                             value=3e-4, format="%.6f", step=1e-4)
            epochs = st.number_input("Max Epochs", min_value=1, max_value=1000, value=50)

        with col2:
            sample_rate = st.selectbox("Sample Rate (Hz)", [8000, 16000, 22050, 44100], index=1)
            max_audio_length = st.number_input("Max Audio Length (s)", min_value=1.0,
                                                max_value=60.0, value=10.0, step=0.5)
            n_mels = st.selectbox("N Mels", [40, 64, 80, 128], index=2)

        with col3:
            seed = st.number_input("Random Seed", min_value=0, value=42)
            early_stop = st.number_input("Early Stop Patience", min_value=1, max_value=100, value=10)
            speaker_independent = st.checkbox("Speaker-Independent Split", value=False)

        run_btn = st.form_submit_button("🚀 Mulai Training", use_container_width=True)

    if run_btn:
        # Bangun command
        cmd = (
            f"python train.py "
            f"--batch_size {batch_size} "
            f"--learning_rate {learning_rate:.6f} "
            f"--epochs {epochs} "
            f"--sample_rate {sample_rate} "
            f"--max_audio_length {max_audio_length} "
            f"--n_mels {n_mels} "
            f"--seed {seed} "
            f"--early_stop_patience {early_stop} "
            + ("--speaker_independent " if speaker_independent else "")
        )

        st.markdown("#### 📋 Command yang akan dijalankan:")
        st.code(cmd, language="bash")

        st.info(
            "⚡ Training berjalan di background.\n\n"
            "Streamlit tidak menampilkan output live untuk proses panjang. "
            "Jalankan command di atas di terminal untuk melihat progress secara penuh.\n\n"
            "Output log tersimpan di `results/training.log`."
        )

        import subprocess, sys
        with st.spinner("Menjalankan training… (bisa memakan waktu lama)"):
            try:
                process = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 jam maksimum
                )
                if process.returncode == 0:
                    st.success("✅ Training selesai!")
                    st.code(process.stdout[-3000:] if len(process.stdout) > 3000
                            else process.stdout)
                else:
                    st.error("❌ Training gagal.")
                    st.code(process.stderr[-3000:] if len(process.stderr) > 3000
                            else process.stderr)
            except subprocess.TimeoutExpired:
                st.warning("Training masih berjalan. Pantau via `results/training.log`.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    # ── Riwayat training ─────────────────────────────────────────────
    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        st.markdown("---")
        st.markdown("#### 📉 Riwayat Training Terakhir")
        with open(history_path, encoding="utf-8") as f:
            hist_data = json.load(f)
        hist = hist_data.get("history", {})
        best_val = hist_data.get("best_val_loss", None)

        if best_val is not None:
            st.metric("Best Validation Loss", f"{best_val:.4f}")

        if hist.get("train_loss") and hist.get("val_loss"):
            chart_df = pd.DataFrame({
                "Train Loss": hist["train_loss"],
                "Val Loss": hist["val_loss"],
            })
            st.line_chart(chart_df)

        if hist.get("learning_rate"):
            lr_df = pd.DataFrame({"Learning Rate": hist["learning_rate"]})
            with st.expander("Learning Rate Schedule"):
                st.line_chart(lr_df)


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Evaluation":

    st.markdown("# 📈 Evaluation")
    st.markdown("Metrik WER dan CER dari model yang sudah dilatih.")
    st.markdown("---")

    if not model_is_trained():
        st.warning(
            "Model belum dilatih. Jalankan training terlebih dahulu.\n\n"
            "CLI: `python train.py`"
        )
        st.stop()

    # ── Pilih split & opsi ───────────────────────────────────────────
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        eval_split = st.selectbox("Split", ["test", "val", "train"], index=0)
    with col_opt2:
        eval_decoding = st.selectbox("Decoding", ["greedy", "beam"], index=0)
    with col_opt3:
        eval_beam = st.slider("Beam Width", 2, 20, 5,
                              disabled=(eval_decoding == "greedy"))

    run_eval = st.button("▶️ Jalankan Evaluasi", type="primary", use_container_width=True)

    # Cek apakah laporan sudah ada
    report_path = RESULTS_DIR / f"evaluation_report_{eval_split}.json"

    if run_eval:
        cmd = (
            f"python evaluate.py "
            f"--split {eval_split} "
            f"--decoding {eval_decoding} "
            f"--beam_width {eval_beam}"
        )
        with st.spinner(f"Mengevaluasi split '{eval_split}'…"):
            import subprocess
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=str(ROOT),
                    capture_output=True, text=True, timeout=7200,
                )
                if proc.returncode == 0:
                    st.success("✅ Evaluasi selesai!")
                else:
                    st.error("❌ Evaluasi gagal.")
                    st.code(proc.stderr[-2000:])
                    st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

    # ── Tampilkan laporan ────────────────────────────────────────────
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("metrics", {})
        wer = metrics.get("wer_pct", None)
        cer = metrics.get("cer_pct", None)
        n_samples_eval = report.get("total_samples", 0)
        avg_conf = report.get("avg_confidence", 0)
        avg_lat = report.get("avg_latency_sec", 0)

        st.markdown(f"#### 📊 Metrik — Split: `{report.get('split', eval_split).upper()}`")

        m1, m2, m3, m4, m5 = st.columns(5)
        if wer is not None and wer >= 0:
            m1.metric("WER", f"{wer:.2f}%",
                      help="Word Error Rate — lebih rendah = lebih baik")
            m2.metric("CER", f"{cer:.2f}%",
                      help="Character Error Rate — lebih rendah = lebih baik")
        else:
            m1.metric("WER", "N/A")
            m2.metric("CER", "N/A")
        m3.metric("Sampel", f"{n_samples_eval}")
        m4.metric("Avg Confidence", f"{avg_conf:.3f}")
        m5.metric("Avg Latency", f"{avg_lat:.3f} s")

        if wer is not None and wer > 90:
            st.warning(
                "⚠️ WER sangat tinggi. Ini wajar jika:\n"
                "- Dataset Bahasa Bengkulu belum tersedia\n"
                "- Model belum cukup dilatih (tambah epoch atau data)\n"
                "- Kualitas audio rekaman rendah"
            )
        elif wer is not None and wer < 30:
            st.success(f"✅ WER {wer:.1f}% — performa model cukup baik!")

        # ── Sampel prediksi ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔍 Sampel Prediksi")
        preds = report.get("all_predictions", [])
        if preds:
            df_preds = pd.DataFrame(preds)
            # Tampilkan kolom penting saja
            show_cols = [c for c in ["file_name", "reference", "hypothesis", "wer", "cer", "confidence"]
                         if c in df_preds.columns]
            st.dataframe(df_preds[show_cols].head(20), use_container_width=True)

            # Download CSV
            samples_csv = RESULTS_DIR / f"sample_predictions_{eval_split}.csv"
            if samples_csv.exists():
                with open(samples_csv, "rb") as f:
                    st.download_button(
                        "⬇️ Download Sample Predictions CSV",
                        data=f.read(),
                        file_name=f"predictions_{eval_split}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Tidak ada data prediksi.")

        # ── Info model ───────────────────────────────────────────────
        model_info = report.get("model_info", {})
        if model_info:
            st.markdown("---")
            st.markdown("#### 🤖 Info Model")
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.json({
                    "Arsitektur": model_info.get("architecture"),
                    "Vocab Size": model_info.get("vocab_size"),
                    "Parameter": f"{model_info.get('parameters', 0):,}",
                })
            with col_i2:
                st.json({
                    "N Mels": model_info.get("n_mels"),
                    "Feature Type": model_info.get("feature_type"),
                    "Sample Rate": model_info.get("sample_rate"),
                    "Max Duration": f"{model_info.get('max_duration_sec')} s",
                })
    else:
        st.info(
            f"Laporan evaluasi untuk split '{eval_split}' belum tersedia.\n\n"
            "Klik **Jalankan Evaluasi** untuk menghitung WER/CER."
        )

        # Tetap tampilkan command alternatif
        st.markdown("**Atau jalankan via CLI:**")
        st.code(f"python evaluate.py --split {eval_split}", language="bash")
