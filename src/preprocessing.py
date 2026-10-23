"""
src/preprocessing.py
====================
Pipeline preprocessing audio untuk Speech-to-Text Bahasa Bengkulu.

Tahapan preprocessing:
1.  Load audio  — membaca file WAV/FLAC/MP3 menggunakan torchaudio
2.  Resample    — konversi sample rate ke target (default 16000 Hz)
3.  Mono        — konversi stereo → mono dengan rata-rata channel
4.  Normalize   — normalisasi amplitudo ke rentang [-1, 1] (peak normalization)
5.  Trim silence— potong keheningan di awal/akhir berdasarkan threshold energi
6.  Pad/Truncate— seragamkan panjang sinyal ke max_length_sec
7.  Feature extraction:
      - Log-Mel Spectrogram  (default, digunakan untuk model CNN/CTC)
      - MFCC                 (opsional, untuk baseline sederhana)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Kelas utama preprocessing audio.

    Parameters
    ----------
    target_sample_rate : int
        Sample rate target. Default: 16000 Hz (standar ASR).
    max_duration_sec : float
        Durasi maksimum audio dalam detik. Audio lebih panjang akan dipotong.
        Default: 10.0 detik.
    n_mels : int
        Jumlah filter bank Mel. Default: 80.
    n_fft : int
        Ukuran FFT window. Default: 512.
    hop_length : int
        Hop length (stride) STFT dalam sampel. Default: 160 (10 ms pada 16 kHz).
    win_length : int
        Panjang window STFT. Default: 400 (25 ms pada 16 kHz).
    n_mfcc : int
        Jumlah koefisien MFCC jika menggunakan MFCC. Default: 40.
    silence_top_db : float
        Ambang keheningan dalam dB untuk trimming. Default: 40 dB.
    feature_type : str
        Tipe fitur: 'logmel' atau 'mfcc'. Default: 'logmel'.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_duration_sec: float = 10.0,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mfcc: int = 40,
        silence_top_db: float = 40.0,
        feature_type: str = "logmel",
    ):
        self.target_sample_rate = target_sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_length_samples = int(target_sample_rate * max_duration_sec)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mfcc = n_mfcc
        self.silence_top_db = silence_top_db
        self.feature_type = feature_type.lower()

        if self.feature_type not in ("logmel", "mfcc"):
            raise ValueError(f"feature_type harus 'logmel' atau 'mfcc', dapat: {feature_type}")

        # Bangun transform torchaudio
        self._mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self._amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        self._mfcc_transform = T.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
            },
        )

        logger.info(
            "AudioPreprocessor diinisialisasi: sr=%d, max_dur=%.1fs, "
            "feature=%s, n_mels=%d",
            target_sample_rate,
            max_duration_sec,
            feature_type,
            n_mels,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load file audio menggunakan torchaudio.

        Returns
        -------
        waveform : torch.Tensor, shape [channels, samples]
        sample_rate : int
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:
            raise RuntimeError(f"Gagal load audio {audio_path}: {exc}") from exc
        logger.debug("Load audio: %s | sr=%d | shape=%s", audio_path.name, sample_rate, waveform.shape)
        return waveform, sample_rate

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample waveform ke target_sample_rate jika berbeda."""
        if orig_sr == self.target_sample_rate:
            return waveform
        resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sample_rate)
        waveform = resampler(waveform)
        logger.debug("Resample: %d → %d Hz", orig_sr, self.target_sample_rate)
        return waveform

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Konversi multi-channel ke mono dengan rata-rata semua channel."""
        if waveform.shape[0] == 1:
            return waveform
        mono = waveform.mean(dim=0, keepdim=True)
        logger.debug("Konversi ke mono dari %d channel", waveform.shape[0])
        return mono

    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Peak normalization: bagi dengan nilai absolut maksimum.
        Menghindari pembagian dengan nol jika sinyal silent.
        """
        peak = waveform.abs().max()
        if peak > 1e-8:
            waveform = waveform / peak
        return waveform

    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Potong keheningan di awal dan akhir sinyal menggunakan
        ambang energi berbasis top_db (implementasi manual tanpa librosa
        agar tidak ada dependensi tambahan pada inference ringan).
        """
        # Hitung energi per frame (window sederhana)
        frame_length = 1024
        hop = 256
        signal = waveform.squeeze(0).numpy()
        energy = np.array([
            np.sum(signal[i : i + frame_length] ** 2)
            for i in range(0, len(signal) - frame_length, hop)
        ])

        if energy.max() < 1e-10:
            # Sinyal hampir silent, kembalikan apa adanya
            return waveform

        db = 10.0 * np.log10(energy / (energy.max() + 1e-10) + 1e-10)
        threshold = -self.silence_top_db

        non_silent = np.where(db > threshold)[0]
        if len(non_silent) == 0:
            return waveform

        start_sample = non_silent[0] * hop
        end_sample = min(non_silent[-1] * hop + frame_length, len(signal))
        trimmed = torch.from_numpy(signal[start_sample:end_sample]).unsqueeze(0)
        logger.debug(
            "Trim silence: %d → %d sampel", len(signal), trimmed.shape[-1]
        )
        return trimmed

    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Seragamkan panjang sinyal ke max_length_samples.
        - Jika lebih panjang: potong (truncate) dari awal.
        - Jika lebih pendek: pad dengan nol di kanan.
        """
        n = waveform.shape[-1]
        if n >= self.max_length_samples:
            waveform = waveform[..., : self.max_length_samples]
        else:
            pad_amount = self.max_length_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ekstrak fitur dari waveform mono yang sudah dipreproses.

        Returns
        -------
        features : torch.Tensor
            - logmel : shape [1, n_mels, time_frames]
            - mfcc   : shape [1, n_mfcc, time_frames]
        """
        if self.feature_type == "logmel":
            mel = self._mel_spectrogram(waveform)      # [1, n_mels, T]
            features = self._amplitude_to_db(mel)      # log scale
        else:  # mfcc
            features = self._mfcc_transform(waveform)  # [1, n_mfcc, T]

        logger.debug("Feature shape: %s", features.shape)
        return features

    def process(
        self,
        audio_path: Union[str, Path],
        trim: bool = True,
        pad_truncate: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Pipeline lengkap: load → resample → mono → normalize
                          → (trim) → (pad/truncate) → extract features.

        Parameters
        ----------
        audio_path : str | Path
        trim       : bool — aktifkan trimming silence (default True)
        pad_truncate : bool — aktifkan padding/truncation (default True)

        Returns
        -------
        features : torch.Tensor   — fitur siap masuk model
        info     : dict           — metadata (sr, duration, shape, dll.)
        """
        waveform, orig_sr = self.load(audio_path)

        duration_orig = waveform.shape[-1] / orig_sr
        waveform = self.resample(waveform, orig_sr)
        waveform = self.to_mono(waveform)
        waveform = self.normalize(waveform)

        if trim:
            waveform = self.trim_silence(waveform)

        duration_processed = waveform.shape[-1] / self.target_sample_rate

        if pad_truncate:
            waveform = self.pad_or_truncate(waveform)

        features = self.extract_features(waveform)

        info = {
            "original_sample_rate": orig_sr,
            "target_sample_rate": self.target_sample_rate,
            "duration_original_sec": round(duration_orig, 4),
            "duration_processed_sec": round(duration_processed, 4),
            "feature_type": self.feature_type,
            "feature_shape": list(features.shape),
        }

        return features, info

    def process_waveform_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
        trim: bool = True,
        pad_truncate: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Proses audio dari bytes (digunakan oleh Streamlit upload).
        Menulis file sementara ke tempfile lalu memanggil process().
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            result = self.process(tmp_path, trim=trim, pad_truncate=pad_truncate)
        finally:
            tmp_path.unlink(missing_ok=True)

        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_duration(audio_path: Union[str, Path]) -> float:
        """Kembalikan durasi file audio dalam detik tanpa load penuh."""
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate

    @staticmethod
    def get_sample_rate(audio_path: Union[str, Path]) -> int:
        """Kembalikan sample rate file audio."""
        info = torchaudio.info(str(audio_path))
        return info.sample_rate

    def __repr__(self) -> str:
        return (
            f"AudioPreprocessor(sr={self.target_sample_rate}, "
            f"max_dur={self.max_duration_sec}s, "
            f"feature={self.feature_type}, "
            f"n_mels={self.n_mels})"
        )
