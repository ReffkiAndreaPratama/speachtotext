"""
src/dataset.py
==============
Dataset loader dan manajemen data untuk Speech-to-Text Bahasa Bengkulu.

Kelas utama:
    - CharacterTokenizer   : encoder/decoder karakter ↔ indeks integer
    - BengkuluSpeechDataset: torch.utils.data.Dataset untuk training/eval
    - DatasetManager       : membaca metadata.csv, split train/val/test,
                             dan menyediakan DataLoader
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)


# ===========================================================================
# Karakter Tokenizer
# ===========================================================================

class CharacterTokenizer:
    """
    Tokenizer level karakter.

    Token spesial:
        <pad>  — padding (indeks 0)
        <sos>  — start-of-sequence (indeks 1)
        <eos>  — end-of-sequence (indeks 2)
        <unk>  — unknown character (indeks 3)

    Parameters
    ----------
    vocab : list[str] | None
        Daftar karakter yang akan digunakan. Jika None, vocab dibangun
        otomatis dari dataset melalui build_vocab().
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, vocab: Optional[List[str]] = None):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        if vocab is not None:
            self._build_from_vocab(vocab)

    def _build_from_vocab(self, vocab: List[str]) -> None:
        all_tokens = self.SPECIAL_TOKENS + sorted(set(vocab))
        self.char2idx = {ch: idx for idx, ch in enumerate(all_tokens)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def build_vocab(self, texts: List[str]) -> None:
        """Bangun vocab dari list teks transkripsi."""
        chars = set()
        for text in texts:
            chars.update(text)
        self._build_from_vocab(list(chars))
        logger.info("Vocab dibangun: %d karakter unik + %d token spesial = %d total",
                    len(chars), len(self.SPECIAL_TOKENS), len(self.char2idx))

    def encode(self, text: str, add_sos_eos: bool = True) -> List[int]:
        """Ubah teks menjadi list indeks integer."""
        unk_idx = self.char2idx[self.UNK_TOKEN]
        indices = [self.char2idx.get(ch, unk_idx) for ch in text]
        if add_sos_eos:
            indices = [self.char2idx[self.SOS_TOKEN]] + indices + [self.char2idx[self.EOS_TOKEN]]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Ubah list indeks kembali menjadi teks."""
        special_set = set(self.char2idx[t] for t in self.SPECIAL_TOKENS)
        chars = []
        for idx in indices:
            if remove_special and idx in special_set:
                continue
            chars.append(self.idx2char.get(idx, self.UNK_TOKEN))
        return "".join(chars)

    def save(self, path: Path) -> None:
        """Simpan vocab ke file teks (satu karakter per baris)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for idx in sorted(self.idx2char.keys()):
                f.write(f"{idx}\t{repr(self.idx2char[idx])}\n")
        logger.info("Vocab disimpan ke %s", path)

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        """Load vocab dari file yang disimpan dengan save()."""
        path = Path(path)
        idx2char: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                idx = int(parts[0])
                char = parts[1].strip("'\"")
                # eval repr untuk karakter spesial seperti \n, spasi
                try:
                    char = bytes(char, "utf-8").decode("unicode_escape")
                except Exception:
                    pass
                idx2char[idx] = char
        tokenizer = cls()
        tokenizer.idx2char = idx2char
        tokenizer.char2idx = {ch: idx for idx, ch in idx2char.items()}
        logger.info("Vocab di-load dari %s (%d token)", path, len(tokenizer.char2idx))
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    @property
    def pad_idx(self) -> int:
        return self.char2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.char2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.char2idx[self.EOS_TOKEN]

    def __repr__(self) -> str:
        return f"CharacterTokenizer(vocab_size={self.vocab_size})"


# ===========================================================================
# Dataset
# ===========================================================================

class BengkuluSpeechDataset(Dataset):
    """
    torch.utils.data.Dataset untuk dataset speech Bahasa Bengkulu.

    Setiap item mengembalikan:
        features  : torch.Tensor — fitur audio [n_mels, T]
        targets   : torch.Tensor — indeks karakter (encoded)
        input_len : int          — panjang time frames fitur
        target_len: int          — panjang sekuens target

    Parameters
    ----------
    metadata   : pd.DataFrame  — kolom: file_name, text, speaker_id
    audio_dir  : Path          — direktori tempat file audio berada
    preprocessor : AudioPreprocessor
    tokenizer  : CharacterTokenizer
    augment    : bool          — aktifkan augmentasi (hanya saat training)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_dir: Path,
        preprocessor: AudioPreprocessor,
        tokenizer: CharacterTokenizer,
        augment: bool = False,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.augment = augment

        # Augmentasi ringan — SpecAugment (time & freq masking)
        if augment:
            import torchaudio.transforms as T
            self._time_mask = T.TimeMasking(time_mask_param=30)
            self._freq_mask = T.FrequencyMasking(freq_mask_param=15)

        logger.info("Dataset diinisialisasi: %d sampel, augment=%s",
                    len(self.metadata), augment)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.metadata.iloc[idx]
        audio_path = self.audio_dir / row["file_name"]

        # --- Feature extraction ---
        try:
            features, _ = self.preprocessor.process(audio_path)
        except Exception as exc:
            logger.warning("Gagal proses audio %s: %s. Skip ke item berikutnya.", audio_path, exc)
            # Kembalikan tensor kosong; collate_fn akan filter ini
            dummy = torch.zeros(1, self.preprocessor.n_mels, 10)
            return dummy.squeeze(0), torch.tensor([0], dtype=torch.long), 10, 1

        # features shape: [1, n_mels, T]
        features = features.squeeze(0)  # → [n_mels, T]

        # --- Augmentasi (hanya training) ---
        if self.augment:
            features = self._freq_mask(features.unsqueeze(0)).squeeze(0)
            features = self._time_mask(features.unsqueeze(0)).squeeze(0)

        # --- Tokenize ---
        text = str(row["text"]).strip().lower()
        target_indices = self.tokenizer.encode(text, add_sos_eos=True)
        targets = torch.tensor(target_indices, dtype=torch.long)

        input_len = features.shape[-1]   # time frames
        target_len = len(targets)

        return features, targets, input_len, target_len

    def get_speaker_list(self) -> List[str]:
        """Kembalikan list unik speaker_id."""
        return self.metadata["speaker_id"].unique().tolist()


# ===========================================================================
# Collate Function
# ===========================================================================

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate untuk padding batch:
    - features  : [B, n_mels, T_max]   (pad kanan dengan 0)
    - targets   : [B, S_max]           (pad kanan dengan pad_idx=0)
    - input_lens: [B]
    - target_lens:[B]
    """
    features, targets, input_lens, target_lens = zip(*batch)

    # Pad features (dim=-1 adalah dimensi waktu)
    max_t = max(f.shape[-1] for f in features)
    padded_features = torch.zeros(len(features), features[0].shape[0], max_t)
    for i, feat in enumerate(features):
        padded_features[i, :, : feat.shape[-1]] = feat

    # Pad targets
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    input_lens_tensor = torch.tensor(input_lens, dtype=torch.long)
    target_lens_tensor = torch.tensor(target_lens, dtype=torch.long)

    return padded_features, padded_targets, input_lens_tensor, target_lens_tensor


# ===========================================================================
# Dataset Manager
# ===========================================================================

class DatasetManager:
    """
    Membaca metadata.csv, membangun tokenizer, melakukan split
    train/val/test, dan menyediakan DataLoader.

    Format metadata.csv:
        file_name,text,speaker_id

    Parameters
    ----------
    dataset_dir    : Path — root dataset (berisi audio/ dan metadata.csv)
    preprocessor   : AudioPreprocessor
    train_ratio    : float — proporsi training (default 0.8)
    val_ratio      : float — proporsi validasi (default 0.1)
    test_ratio     : float — proporsi test (default 0.1)
    speaker_independent : bool
        Jika True, split dilakukan per speaker sehingga speaker pada test
        tidak muncul di training (evaluasi speaker-independent).
        Jika False, split acak per sampel.
    seed           : int — random seed untuk reproducibility
    """

    METADATA_FILE = "metadata.csv"
    VOCAB_FILE = "models/vocab.txt"

    def __init__(
        self,
        dataset_dir: Path,
        preprocessor: AudioPreprocessor,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        speaker_independent: bool = False,
        seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audio"
        self.preprocessor = preprocessor
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.speaker_independent = speaker_independent
        self.seed = seed

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio harus = 1.0")

        self.tokenizer: Optional[CharacterTokenizer] = None
        self._metadata: Optional[pd.DataFrame] = None

        logger.info(
            "DatasetManager: dir=%s, split=%.0f/%.0f/%.0f, speaker_indep=%s",
            dataset_dir,
            train_ratio * 100,
            val_ratio * 100,
            test_ratio * 100,
            speaker_independent,
        )

    # ------------------------------------------------------------------
    # Load & split
    # ------------------------------------------------------------------

    def load_metadata(self) -> pd.DataFrame:
        """Baca dan validasi metadata.csv."""
        csv_path = self.dataset_dir / self.METADATA_FILE
        if not csv_path.exists():
            raise FileNotFoundError(
                f"metadata.csv tidak ditemukan di {self.dataset_dir}. "
                "Jalankan prepare_dataset.py terlebih dahulu."
            )
        df = pd.read_csv(csv_path, dtype=str)
        required_cols = {"file_name", "text", "speaker_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Kolom yang hilang di metadata.csv: {missing}")

        # Bersihkan data
        df = df.dropna(subset=list(required_cols))
        df["text"] = df["text"].str.strip().str.lower()
        df = df[df["text"].str.len() > 0]
        df = df.reset_index(drop=True)

        self._metadata = df
        logger.info("Metadata dimuat: %d baris valid", len(df))
        return df

    def build_tokenizer(self, df: Optional[pd.DataFrame] = None) -> CharacterTokenizer:
        """Bangun CharacterTokenizer dari teks transkripsi."""
        if df is None:
            if self._metadata is None:
                self.load_metadata()
            df = self._metadata
        texts = df["text"].tolist()
        tokenizer = CharacterTokenizer()
        tokenizer.build_vocab(texts)
        self.tokenizer = tokenizer
        return tokenizer

    def split(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Bagi dataframe menjadi train/val/test sesuai strategi.

        Returns
        -------
        df_train, df_val, df_test : pd.DataFrame
        """
        if df is None:
            if self._metadata is None:
                self.load_metadata()
            df = self._metadata

        rng = random.Random(self.seed)

        if self.speaker_independent:
            return self._split_by_speaker(df, rng)
        else:
            return self._split_random(df, rng)

    def _split_random(
        self, df: pd.DataFrame, rng: random.Random
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split acak per sampel."""
        indices = list(range(len(df)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        logger.info(
            "Split acak: train=%d, val=%d, test=%d",
            len(df_train), len(df_val), len(df_test),
        )
        return df_train, df_val, df_test

    def _split_by_speaker(
        self, df: pd.DataFrame, rng: random.Random
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split speaker-independent:
        - Shuffle list speaker, lalu bagi sesuai rasio.
        - Semua sampel dari speaker test tidak ada di training.
        """
        speakers = df["speaker_id"].unique().tolist()
        rng.shuffle(speakers)

        n_sp = len(speakers)
        n_train_sp = max(1, int(n_sp * self.train_ratio))
        n_val_sp = max(1, int(n_sp * self.val_ratio))

        train_speakers = set(speakers[:n_train_sp])
        val_speakers = set(speakers[n_train_sp : n_train_sp + n_val_sp])
        test_speakers = set(speakers[n_train_sp + n_val_sp :])

        # Jika hanya ada sedikit speaker, pastikan test tidak kosong
        if not test_speakers:
            test_speakers = {speakers[-1]}
            train_speakers.discard(speakers[-1])

        df_train = df[df["speaker_id"].isin(train_speakers)].reset_index(drop=True)
        df_val = df[df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
        df_test = df[df["speaker_id"].isin(test_speakers)].reset_index(drop=True)

        logger.info(
            "Split speaker-independent: "
            "train=%d sampel (%d speaker), val=%d sampel (%d speaker), "
            "test=%d sampel (%d speaker)",
            len(df_train), len(train_speakers),
            len(df_val), len(val_speakers),
            len(df_test), len(test_speakers),
        )
        return df_train, df_val, df_test

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Siapkan DataLoader untuk train, val, dan test.

        Parameters
        ----------
        batch_size  : int
        num_workers : int — set 0 di Windows untuk menghindari multiprocessing issue

        Returns
        -------
        train_loader, val_loader, test_loader
        """
        df = self._metadata if self._metadata is not None else self.load_metadata()

        # Split SEBELUM membangun tokenizer untuk menghindari leakage info
        # karakter dari test set ke dalam vocab training.
        # Vocab dibangun hanya dari training set.
        df_train, df_val, df_test = self.split(df)

        if self.tokenizer is None:
            self.build_tokenizer(df_train)  # HANYA dari training set

        train_dataset = BengkuluSpeechDataset(
            df_train, self.audio_dir, self.preprocessor, self.tokenizer, augment=True
        )
        val_dataset = BengkuluSpeechDataset(
            df_val, self.audio_dir, self.preprocessor, self.tokenizer, augment=False
        )
        test_dataset = BengkuluSpeechDataset(
            df_test, self.audio_dir, self.preprocessor, self.tokenizer, augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "DataLoader siap: train=%d batch, val=%d batch, test=%d batch (batch_size=%d)",
            len(train_loader), len(val_loader), len(test_loader), batch_size,
        )
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # Statistik dataset
    # ------------------------------------------------------------------

    def compute_statistics(self) -> Dict:
        """
        Hitung statistik dataset dari metadata dan file audio.
        Digunakan oleh halaman Dataset Overview di Streamlit.
        """
        if self._metadata is None:
            self.load_metadata()
        df = self._metadata

        total_samples = len(df)
        unique_speakers = df["speaker_id"].nunique()
        speaker_counts = df["speaker_id"].value_counts().to_dict()

        durations = []
        missing_files = []
        for _, row in df.iterrows():
            audio_path = self.audio_dir / row["file_name"]
            if audio_path.exists():
                try:
                    dur = AudioPreprocessor.get_duration(audio_path)
                    durations.append(dur)
                except Exception:
                    durations.append(0.0)
            else:
                missing_files.append(row["file_name"])
                durations.append(0.0)

        total_duration_sec = sum(durations)
        avg_duration_sec = total_duration_sec / max(len(durations), 1)

        stats = {
            "total_samples": total_samples,
            "unique_speakers": unique_speakers,
            "speaker_counts": speaker_counts,
            "total_duration_sec": round(total_duration_sec, 2),
            "total_duration_min": round(total_duration_sec / 60, 2),
            "avg_duration_sec": round(avg_duration_sec, 2),
            "min_duration_sec": round(min(durations), 2) if durations else 0.0,
            "max_duration_sec": round(max(durations), 2) if durations else 0.0,
            "missing_files": missing_files,
            "durations": durations,
        }
        return stats
