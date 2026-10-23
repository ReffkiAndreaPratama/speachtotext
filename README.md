# 🎙️ Speech-to-Text Bahasa Bengkulu

> Prototype sistem pengenalan ucapan berbahasa daerah Bengkulu menggunakan Deep Learning (CNN-BiLSTM-CTC)

**Informatics Practitioner Lecture Series**
Tema: *Bridging Global Infrastructure to Address Local Challenge with HPC*
Pemateri: Asst. Prof. Dr. Worawan Diaz Carballo — Thammasat University

---

## 📋 Daftar Isi

1. [Deskripsi Project](#1-deskripsi-project)
2. [Latar Belakang](#2-latar-belakang)
3. [Hubungan dengan Materi HPC](#3-hubungan-dengan-materi-hpc)
4. [Konsep Integrator](#4-konsep-integrator)
5. [Permasalahan](#5-permasalahan)
6. [Tujuan](#6-tujuan)
7. [Dataset](#7-dataset)
8. [Format Dataset](#8-format-dataset)
9. [Preprocessing Audio](#9-preprocessing-audio)
10. [Arsitektur Model](#10-arsitektur-model)
11. [Training](#11-training)
12. [Evaluasi](#12-evaluasi)
13. [Word Error Rate (WER)](#13-word-error-rate-wer)
14. [Character Error Rate (CER)](#14-character-error-rate-cer)
15. [Instalasi](#15-instalasi)
16. [Menyiapkan Dataset](#16-menyiapkan-dataset)
17. [Cara Training](#17-cara-training)
18. [Cara Evaluasi](#18-cara-evaluasi)
19. [Menjalankan Aplikasi](#19-menjalankan-aplikasi)
20. [Struktur Folder](#20-struktur-folder)
21. [Limitasi Penelitian](#21-limitasi-penelitian)
22. [Pengembangan Selanjutnya](#22-pengembangan-selanjutnya)

---

## 1. Deskripsi Project

Project ini membangun **prototype sistem Speech-to-Text (STT) untuk Bahasa Daerah Bengkulu** menggunakan teknik Deep Learning. Sistem ini menerima input rekaman audio berbahasa Bengkulu dan menghasilkan output berupa teks transkripsi.

Sistem dibangun sebagai studi kasus dari materi kuliah tentang pemanfaatan High Performance Computing (HPC) dan kecerdasan buatan (AI) untuk menjawab tantangan lokal — sebuah pendekatan yang digagas oleh Asst. Prof. Dr. Worawan Diaz Carballo dari Thammasat University.

**Status saat ini:** Pipeline lengkap tersedia. Transkripsi bermakna akan tersedia setelah dataset Bahasa Bengkulu dikumpulkan dan model dilatih.

---

## 2. Latar Belakang

Indonesia memiliki lebih dari 700 bahasa daerah, namun hampir seluruhnya tidak terwakili dalam sistem pengenalan ucapan (ASR) modern. Bahasa Bengkulu adalah salah satu bahasa daerah yang digunakan oleh masyarakat Provinsi Bengkulu, Sumatera.

Sementara itu, perkembangan teknologi AI dan HPC membuka peluang untuk membangun sistem STT bagi bahasa-bahasa minoritas tanpa harus mengandalkan infrastruktur komersial berskala besar. Inspirasi langsung datang dari studi kasus yang disampaikan dalam kuliah — sistem penerjemah bahasa minoritas Hmong — yang menunjukkan bahwa AI dapat digunakan untuk melestarikan dan memberdayakan bahasa lokal.

---

## 3. Hubungan dengan Materi HPC

Materi kuliah *"Bridging Global Infrastructure to Address Local Challenge with HPC"* membahas:

| Topik HPC | Relevansi dengan Project Ini |
|-----------|------------------------------|
| High Performance Computing | Training model deep learning membutuhkan GPU; HPC mempercepat proses ini dari hari ke menit |
| Parallel Computing | PyTorch menggunakan CUDA untuk komputasi paralel pada GPU |
| GPU Computing | Model CNN-BiLSTM-CTC secara otomatis menggunakan CUDA jika tersedia |
| Data dalam jumlah besar | Dataset audio speech bisa mencapai ratusan GB; HPC diperlukan untuk skala penuh |
| Linux & SSH | Training skala besar dijalankan di server Linux via SSH (HPC cluster) |
| AI untuk permasalahan lokal | Sistem STT ini dirancang spesifik untuk Bahasa Bengkulu |
| Speech-to-text | Core fitur utama project ini |

---

## 4. Konsep Integrator

Dalam konteks materi HPC, **Integrator** adalah entitas yang menjembatani infrastruktur global (cloud, HPC, GPU cluster) dengan kebutuhan lokal yang spesifik. Project ini berperan sebagai integrator dengan cara:

1. **Mengintegrasikan framework global** (PyTorch, torchaudio) ke dalam konteks bahasa daerah Indonesia
2. **Menjembatani gap teknologi** antara kemajuan ASR global dan kelangkaan dukungan bahasa lokal
3. **Menyederhanakan akses** — antarmuka Streamlit memungkinkan pengguna non-teknis menggunakan sistem
4. **Memungkinkan deployment lokal** — sistem dapat berjalan di mesin lokal tanpa ketergantungan API pihak ketiga
5. **Mereplikasi pendekatan Hmong** — seperti yang dicontohkan dalam materi, sistem ini mengadaptasi metodologi ASR untuk bahasa minoritas Indonesia

---

## 5. Permasalahan

- **Tidak ada sistem STT komersial** yang mendukung Bahasa Bengkulu
- **Bahasa daerah terancam punah** karena kurangnya dokumentasi digital
- **Kesenjangan digital** — masyarakat penutur Bahasa Bengkulu tidak dapat menggunakan asisten suara modern
- **Kelangkaan dataset** — tidak ada korpus ucapan Bahasa Bengkulu yang tersedia secara publik
- **Sumber daya terbatas** — penelitian bahasa daerah umumnya kekurangan pendanaan dan infrastruktur

---

## 6. Tujuan

1. Membangun **pipeline Speech-to-Text** yang lengkap dan dapat direproduksi
2. Merancang **format dataset** yang dapat langsung digunakan ketika data dikumpulkan
3. Mengimplementasikan **model Deep Learning** (CNN-BiLSTM-CTC) yang sesuai untuk ASR low-resource
4. Menyediakan **tool validasi dataset** untuk memastikan kualitas data
5. Membangun **antarmuka web** yang mudah digunakan
6. Menghitung **WER dan CER** sebagai metrik evaluasi standar ASR
7. Mendemonstrasikan penerapan **konsep HPC** untuk tantangan lokal

---

## 7. Dataset

> ⚠️ **Penting:** Dataset Bahasa Bengkulu tidak tersedia secara publik. Project ini menyediakan pipeline yang siap menerima dataset ketika data sudah dikumpulkan.

Dataset perlu dikumpulkan secara mandiri dengan cara:

- Merekam penutur asli Bahasa Bengkulu membacakan teks atau berbicara bebas
- Mengumpulkan rekaman audio dari komunitas penutur Bahasa Bengkulu
- Berkolaborasi dengan lembaga bahasa atau universitas di Provinsi Bengkulu

**Rekomendasi minimum untuk eksperimen awal:**
- ≥ 100 sampel audio
- ≥ 3 speaker berbeda (untuk evaluasi yang lebih valid)
- Durasi 1–10 detik per sampel
- Format WAV, sample rate 16000 Hz (atau akan di-resample otomatis)

---

## 8. Format Dataset

```
dataset/
├── audio/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

**Format `metadata.csv`:**

```csv
file_name,text,speaker_id
audio_001.wav,aku pergi ke pasar,speaker_01
audio_002.wav,rumah kami di tepi sungai,speaker_01
audio_003.wav,cuaca hari ini sangat panas,speaker_02
```

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `file_name` | string | Nama file audio (relatif terhadap `dataset/audio/`) |
| `text` | string | Transkripsi teks dalam Bahasa Bengkulu (huruf kecil disarankan) |
| `speaker_id` | string | Identifier unik per speaker (mis. `speaker_01`, `spk_f_001`) |

---

## 9. Preprocessing Audio

Pipeline preprocessing diimplementasikan di `src/preprocessing.py`:

| Tahap | Deskripsi |
|-------|-----------|
| **Load** | Membaca file WAV/FLAC/MP3 menggunakan `torchaudio` |
| **Resample** | Konversi sample rate ke 16000 Hz jika berbeda |
| **Mono** | Konversi stereo → mono (rata-rata channel) |
| **Normalize** | Peak normalization ke rentang [-1, 1] |
| **Trim Silence** | Potong keheningan di awal/akhir berdasarkan threshold energi |
| **Pad/Truncate** | Seragamkan panjang ke `max_duration_sec` (default: 10 detik) |
| **Feature Extraction** | **Log-Mel Spectrogram** (default): [1, 80, T] atau **MFCC**: [1, 40, T] |

**Mengapa Log-Mel Spectrogram?**
Mel Spectrogram mensimulasikan persepsi frekuensi telinga manusia (skala Mel). Konversi ke skala logaritmik (dB) mengurangi rentang dinamis dan membuat fitur lebih stabil untuk training neural network.

---

## 10. Arsitektur Model

**CNN-BiLSTM-CTC** diimplementasikan di `src/model.py`:

```
Input: Log-Mel Spectrogram [B, 80, T]
         │
         ▼
┌─────────────────────────────────┐
│  CNN Encoder (3 blok)           │
│  ConvBlock: Conv2d → BN → ReLU  │
│  channels: 1 → 32 → 64 → 128   │
│  MaxPool hanya pada blok pertama│
└────────────────┬────────────────┘
                 │ [B, 128*freq_reduced, T']
                 ▼
┌─────────────────────────────────┐
│  BiLSTM (3 layer)               │
│  hidden: 256 (per direction)    │
│  output: 512 per time step      │
│  dropout: 0.3                   │
└────────────────┬────────────────┘
                 │ [T', B, 512]
                 ▼
┌─────────────────────────────────┐
│  Dropout (0.3) + Linear         │
│  output: vocab_size             │
└────────────────┬────────────────┘
                 │
                 ▼
            Log Softmax
                 │
                 ▼
         CTC Loss / Decode
```

**Mengapa CNN-BiLSTM-CTC?**
- **CTC (Connectionist Temporal Classification):** tidak memerlukan alignment manual antara audio dan transkripsi — ideal untuk dataset low-resource
- **CNN:** menangkap pola lokal frekuensi-waktu pada spectrogram
- **BiLSTM:** memodelkan konteks temporal jangka panjang dari dua arah
- **Cocok untuk low-resource:** arsitektur ini terbukti bekerja baik dengan data terbatas

---

## 11. Training

Diimplementasikan di `src/trainer.py` dan `train.py`.

| Komponen | Detail |
|----------|--------|
| **Loss** | CTC Loss (`torch.nn.CTCLoss`, blank=0) |
| **Optimizer** | Adam (lr=3e-4, weight_decay=1e-5) |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Early Stopping** | Patience=10 epoch |
| **Gradient Clipping** | max_norm=5.0 |
| **Augmentasi** | SpecAugment (Time Masking + Frequency Masking) |
| **Split** | 80% train / 10% val / 10% test (atau speaker-independent) |
| **Checkpoint** | `models/best_model.pt` (best val loss) dan `last_checkpoint.pt` |
| **Seed** | Dapat dikonfigurasi (default: 42) |

---

## 12. Evaluasi

Diimplementasikan di `evaluate.py` menggunakan library `jiwer`.

Metrik dihitung pada split `test` (default) dengan data yang tidak digunakan selama training.

> **Catatan:** Hasil evaluasi hanya valid jika model sudah dilatih dengan dataset Bahasa Bengkulu nyata. Program tidak mengarang metrik.

---

## 13. Word Error Rate (WER)

$$\text{WER} = \frac{S + D + I}{N}$$

- **S** = jumlah kata yang disubstitusi
- **D** = jumlah kata yang dihapus (deletion)
- **I** = jumlah kata yang disisipkan (insertion)
- **N** = jumlah total kata dalam referensi

WER lebih rendah = performa lebih baik. WER = 0% berarti transkripsi sempurna.

**Interpretasi umum:**
| WER | Kualitas |
|-----|----------|
| < 10% | Sangat baik (setara manusia) |
| 10–30% | Baik (layak pakai dengan post-editing) |
| 30–60% | Cukup (masih bermanfaat) |
| > 60% | Perlu lebih banyak data / training |

---

## 14. Character Error Rate (CER)

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

Sama seperti WER tetapi dihitung per karakter, bukan per kata. CER biasanya lebih rendah dari WER dan lebih sensitif untuk bahasa dengan kata-kata panjang atau sistem penulisan yang berbeda.

---

## 15. Instalasi

### Prasyarat
- Python 3.11 atau lebih baru
- pip

### Langkah instalasi

```bash
# 1. Clone repository
git clone https://github.com/[USERNAME]/speech-to-text-bahasa-bengkulu.git
cd speech-to-text-bahasa-bengkulu

# 2. Buat virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Instalasi PyTorch dengan GPU (CUDA 11.8)

```bash
pip install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Instalasi PyTorch CPU only

```bash
pip install torch==2.2.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

---

## 16. Menyiapkan Dataset

### Langkah 1 — Siapkan rekaman audio

Rekam atau kumpulkan file audio Bahasa Bengkulu. Format yang didukung: WAV (disarankan), FLAC, MP3, OGG.

Salin semua file audio ke:
```
dataset/audio/
```

### Langkah 2 — Isi metadata.csv

Edit file `dataset/metadata.csv`:

```csv
file_name,text,speaker_id
audio_001.wav,aku pergi ke pasar kemarin,speaker_01
audio_002.wav,rumah kami di tepi sungai musi,speaker_01
audio_003.wav,cuaca hari ini sangat panas sekali,speaker_02
```

> ⚠️ Jangan memasukkan data contoh di atas jika bukan rekaman nyata.

### Langkah 3 — Validasi dataset

```bash
python prepare_dataset.py
```

Untuk melihat semua opsi:
```bash
python prepare_dataset.py --help
```

Untuk membersihkan entri bermasalah otomatis:
```bash
python prepare_dataset.py --fix
```

---

## 17. Cara Training

### Training dasar

```bash
python train.py
```

### Training dengan parameter kustom

```bash
python train.py \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --epochs 100 \
    --sample_rate 16000 \
    --max_audio_length 10.0 \
    --seed 42 \
    --early_stop_patience 15
```

### Training speaker-independent

```bash
python train.py --speaker_independent
```

### Melanjutkan training dari checkpoint

```bash
python train.py --resume models/last_checkpoint.pt
```

### Lihat semua opsi

```bash
python train.py --help
```

Model terbaik disimpan di `models/best_model.pt`. Log disimpan di `results/training.log`.

---

## 18. Cara Evaluasi

### Evaluasi pada test set

```bash
python evaluate.py
```

### Evaluasi dengan beam search

```bash
python evaluate.py --decoding beam --beam_width 10
```

### Evaluasi pada validation set

```bash
python evaluate.py --split val
```

### Lihat semua opsi

```bash
python evaluate.py --help
```

Output:
- `results/evaluation_report_test.json` — laporan lengkap JSON
- `results/sample_predictions_test.csv` — sampel prediksi vs referensi

---

## 19. Menjalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser di: **http://localhost:8501**

### Halaman aplikasi

| Halaman | Fungsi |
|---------|--------|
| 🎙️ Speech Recognition | Upload audio → transkripsi → download TXT |
| 📊 Dataset Overview | Statistik dataset: jumlah, speaker, durasi, distribusi |
| 🏋️ Training | Konfigurasi dan jalankan training dari UI web |
| 📈 Evaluation | Tampilkan WER, CER, dan sampel prediksi |

---

## 20. Struktur Folder

```
speech-to-text-bahasa-bengkulu/
│
├── app.py                  # Aplikasi web Streamlit (4 halaman)
├── train.py                # Script training (MODE 1)
├── evaluate.py             # Script evaluasi WER/CER
├── prepare_dataset.py      # Validasi dan persiapan dataset
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi ini
├── .gitignore
│
├── dataset/
│   ├── audio/              # File audio WAV/FLAC/MP3
│   │   └── .gitkeep
│   └── metadata.csv        # Transkripsi dan speaker ID
│
├── models/
│   ├── best_model.pt       # Model terbaik (hasil training)
│   ├── last_checkpoint.pt  # Checkpoint terakhir
│   ├── vocab.txt           # Kosakata karakter
│   └── .gitkeep
│
├── results/
│   ├── training.log        # Log training
│   ├── training_history.json
│   ├── evaluation_report_test.json
│   ├── sample_predictions_test.csv
│   ├── dataset_validation_report.json
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── preprocessing.py    # Pipeline preprocessing audio
    ├── dataset.py          # Dataset loader, tokenizer, DataManager
    ├── model.py            # Arsitektur CNN-BiLSTM-CTC
    ├── trainer.py          # Training loop, checkpoint, early stopping
    └── inference.py        # Engine inferensi, CTC decoder
```

---

## 21. Limitasi Penelitian

1. **Tidak ada dataset publik** — Bahasa Bengkulu tidak memiliki korpus ucapan yang tersedia secara publik. Seluruh dataset harus dikumpulkan secara mandiri.

2. **Model belum dilatih** — Pipeline lengkap tersedia, tetapi tanpa data latih, sistem tidak dapat menghasilkan transkripsi bermakna. Ini adalah kondisi yang jujur dan disengaja — bukan kekurangan implementasi.

3. **Sumber daya komputasi** — Training skala penuh pada dataset besar membutuhkan GPU. Tanpa GPU, training akan sangat lambat (bisa berhari-hari pada CPU).

4. **Evaluasi terbatas** — Dengan dataset kecil, WER/CER yang dilaporkan mungkin tidak representatif. Evaluasi yang valid membutuhkan ratusan sampel test.

5. **Dialek** — Bahasa Bengkulu memiliki variasi dialek. Model saat ini tidak memperhitungkan variasi dialek secara eksplisit.

6. **Noise robustness** — Model tidak dilatih secara khusus untuk kondisi noisy. Rekaman di lingkungan bising mungkin menghasilkan akurasi lebih rendah.

7. **Vocabulary coverage** — Karakter tokenizer level karakter sudah menangani OOV (out-of-vocabulary), tetapi performa optimal membutuhkan vocab yang representatif.

8. **Tidak ada language model** — Decoding hanya berbasis akustik (no LM). Integrasi language model bisa meningkatkan WER secara signifikan.

---

## 22. Pengembangan Selanjutnya

### Jangka Pendek
- [ ] Kumpulkan minimal 1000 sampel audio Bahasa Bengkulu dari penutur asli
- [ ] Perluas ke multiple dialek Bengkulu
- [ ] Tambahkan augmentasi audio: noise injection, speed perturbation, pitch shifting
- [ ] Implementasikan data augmentation menggunakan TTS sintetis sebagai bootstrap

### Jangka Menengah
- [ ] Fine-tuning model **Wav2Vec 2.0** atau **Whisper** (pre-trained multilingual) pada dataset Bengkulu — pendekatan transfer learning lebih efisien untuk low-resource
- [ ] Integrasi n-gram language model untuk meningkatkan akurasi decoding
- [ ] Rekam audio pada kondisi yang lebih beragam (indoor, outdoor, noise)
- [ ] Deploy ke HPC cluster untuk training skala penuh

### Jangka Panjang
- [ ] Publikasi dataset Bahasa Bengkulu sebagai sumber terbuka
- [ ] Integrasi **Text-to-Speech** untuk sistem dua arah
- [ ] Aplikasi mobile untuk pengguna lapangan
- [ ] Ekspansi ke bahasa daerah Bengkulu lainnya (Rejang, Serawai, Enggano)
- [ ] Kolaborasi dengan Balai Bahasa Provinsi Bengkulu untuk validasi linguistik

---

## 📚 Referensi

- Graves, A. et al. (2006). *Connectionist Temporal Classification*. ICML.
- Bahdanau, D. et al. (2016). *End-to-End Attention-Based Large Vocabulary Speech Recognition*. ICASSP.
- Park, D. S. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for ASR*. Interspeech.
- Ardila, R. et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. LREC.
- Pratap, V. et al. (2020). *MMS: Scaling Speech Technology to 1000+ Languages*. arXiv.

---

## 🔗 GitHub

```bash
git init
git add .
git commit -m "Initial speech-to-text implementation"
git branch -M main
git remote add origin [GITHUB_REPOSITORY_URL]
git push -u origin main
```

---

## 📄 Lisensi

MIT License — lihat file `LICENSE` untuk detail.

---

*Dibuat sebagai tugas Informatics Practitioner Lecture Series.*
*Pipeline siap — menunggu dataset Bahasa Bengkulu dari komunitas penutur asli.*
