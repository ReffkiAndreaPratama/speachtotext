"""
src/inference.py
================
Engine inferensi untuk Speech-to-Text Bahasa Bengkulu.

Kelas utama:
    - CTCDecoder       : greedy dan beam search decoder untuk output CTC
    - InferenceEngine  : wrapper lengkap — load model, preprocess audio,
                         decode output, kembalikan transkripsi + metadata

Catatan:
    - Model yang belum dilatih TIDAK akan menghasilkan transkripsi bermakna.
    - Engine ini siap digunakan begitu model sudah dilatih dengan dataset nyata.
    - Confidence score dihitung dari rata-rata probabilitas karakter terprediksi.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.preprocessing import AudioPreprocessor
from src.dataset import CharacterTokenizer
from src.model import SpeechRecognitionModel, ModelConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# CTC Decoder
# ===========================================================================

class CTCDecoder:
    """
    Decoder untuk output CTC model.

    Mendukung dua strategi:
        1. Greedy Decoding   — cepat, argmax per time step
        2. Beam Search       — lebih akurat, lebih lambat

    Parameters
    ----------
    tokenizer  : CharacterTokenizer
    blank_idx  : int — indeks blank token (default 0 = PAD)
    """

    def __init__(self, tokenizer: CharacterTokenizer, blank_idx: int = 0):
        self.tokenizer = tokenizer
        self.blank_idx = blank_idx

    # ------------------------------------------------------------------
    # Greedy Decoding
    # ------------------------------------------------------------------

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Greedy CTC decoding: argmax per time step + collapse repeated + hapus blank.

        Parameters
        ----------
        log_probs     : [T, vocab_size]  — log probabilitas satu sampel
        output_length : int | None       — panjang valid T (potong padding)

        Returns
        -------
        text       : str   — transkripsi
        confidence : float — rata-rata prob karakter non-blank [0, 1]
        """
        if output_length is not None:
            log_probs = log_probs[:output_length]

        # Argmax per time step
        probs = torch.exp(log_probs)               # [T, vocab]
        best_indices = torch.argmax(probs, dim=-1) # [T]
        best_probs = probs[
            torch.arange(len(best_indices)), best_indices
        ]                                          # [T]

        # Collapse CTC: hapus berulang + hapus blank
        decoded_indices = []
        decoded_probs = []
        prev_idx = None
        for idx, prob in zip(best_indices.tolist(), best_probs.tolist()):
            if idx != prev_idx:
                if idx != self.blank_idx:
                    decoded_indices.append(idx)
                    decoded_probs.append(prob)
            prev_idx = idx

        # Decode ke teks
        text = self.tokenizer.decode(decoded_indices, remove_special=True)

        # Confidence: rata-rata probabilitas karakter yang dipilih
        confidence = float(sum(decoded_probs) / max(len(decoded_probs), 1))

        return text, confidence

    # ------------------------------------------------------------------
    # Beam Search Decoding
    # ------------------------------------------------------------------

    def beam_search_decode(
        self,
        log_probs: torch.Tensor,
        output_length: Optional[int] = None,
        beam_width: int = 5,
    ) -> Tuple[str, float]:
        """
        Beam search CTC decoding.

        Implementasi sederhana tanpa language model (pure acoustic).

        Parameters
        ----------
        log_probs    : [T, vocab_size]
        output_length: int | None
        beam_width   : int — jumlah hypothesis yang dijaga

        Returns
        -------
        text       : str
        confidence : float — normalized log-prob terbaik
        """
        if output_length is not None:
            log_probs = log_probs[:output_length]

        T, vocab_size = log_probs.shape
        probs = torch.exp(log_probs).cpu().numpy()

        # Beam: list (sequence, last_char, cumulative_log_prob)
        import math
        NEG_INF = -math.inf

        # Inisialisasi beam dengan sequence kosong
        # beams: dict { (sequence_tuple, last_char) : log_prob }
        beams = {((), self.blank_idx): 0.0}

        for t in range(T):
            new_beams: dict = {}

            for (seq, last_char), log_prob in beams.items():
                for c in range(vocab_size):
                    p = float(probs[t, c])
                    if p < 1e-10:
                        continue
                    lp_c = math.log(p)

                    if c == self.blank_idx:
                        # Blank: sequence tidak berubah
                        key = (seq, self.blank_idx)
                        new_log_prob = log_prob + lp_c
                        if key not in new_beams or new_beams[key] < new_log_prob:
                            new_beams[key] = new_log_prob
                    elif c == last_char:
                        # Sama dengan karakter terakhir: hanya bisa ditambah setelah blank
                        key = (seq, c)
                        new_log_prob = log_prob + lp_c
                        if key not in new_beams or new_beams[key] < new_log_prob:
                            new_beams[key] = new_log_prob
                    else:
                        # Karakter baru
                        new_seq = seq + (c,)
                        key = (new_seq, c)
                        new_log_prob = log_prob + lp_c
                        if key not in new_beams or new_beams[key] < new_log_prob:
                            new_beams[key] = new_log_prob

            # Pruning: ambil top beam_width
            if new_beams:
                sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                beams = dict(sorted_beams[:beam_width])
            else:
                break

        # Ambil hypothesis terbaik
        if not beams:
            return "", 0.0

        best_key, best_log_prob = max(beams.items(), key=lambda x: x[1])
        best_seq = list(best_key[0])

        text = self.tokenizer.decode(best_seq, remove_special=True)

        # Normalized confidence
        seq_len = max(len(best_seq), 1)
        normalized_log_prob = best_log_prob / seq_len
        confidence = float(math.exp(max(normalized_log_prob, -10)))
        confidence = min(max(confidence, 0.0), 1.0)

        return text, confidence


# ===========================================================================
# Inference Engine
# ===========================================================================

class InferenceEngine:
    """
    Engine inferensi lengkap: load model + preprocess + decode.

    Cara penggunaan:
        engine = InferenceEngine.from_checkpoint(
            model_path="models/best_model.pt",
            vocab_path="models/vocab.txt",
        )
        result = engine.transcribe("audio/rekaman.wav")
        print(result["text"])

    Parameters
    ----------
    model        : SpeechRecognitionModel (sudah di-load, eval mode)
    preprocessor : AudioPreprocessor
    tokenizer    : CharacterTokenizer
    decoder      : CTCDecoder
    device       : torch.device
    """

    def __init__(
        self,
        model: SpeechRecognitionModel,
        preprocessor: AudioPreprocessor,
        tokenizer: CharacterTokenizer,
        decoder: CTCDecoder,
        device: torch.device,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.device = device

        self.model.eval()
        logger.info(
            "InferenceEngine siap | device=%s | vocab_size=%d",
            device, tokenizer.vocab_size,
        )

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        vocab_path: Union[str, Path],
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        max_duration_sec: float = 10.0,
        n_mels: int = 80,
        feature_type: str = "logmel",
        decoding_strategy: str = "greedy",
    ) -> "InferenceEngine":
        """
        Buat InferenceEngine dari file checkpoint dan vocab.

        Parameters
        ----------
        model_path         : Path ke file .pt model
        vocab_path         : Path ke file vocab.txt (output CharacterTokenizer.save)
        device             : torch.device — default: cuda jika tersedia
        sample_rate        : target sample rate (harus sama dengan training)
        max_duration_sec   : durasi maksimum (harus sama dengan training)
        n_mels             : jumlah Mel filter (harus sama dengan training)
        feature_type       : 'logmel' atau 'mfcc' (harus sama dengan training)
        decoding_strategy  : 'greedy' atau 'beam'

        Returns
        -------
        InferenceEngine
        """
        model_path = Path(model_path)
        vocab_path = Path(vocab_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info("Loading model dari %s ...", model_path)
        model, extra = SpeechRecognitionModel.load(model_path, device=device)

        # Load tokenizer
        logger.info("Loading vocab dari %s ...", vocab_path)
        tokenizer = CharacterTokenizer.load(vocab_path)

        # Preprocessor — gunakan config yang sama dengan training
        preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            max_duration_sec=max_duration_sec,
            n_mels=n_mels,
            feature_type=feature_type,
        )

        # Decoder
        decoder = CTCDecoder(tokenizer, blank_idx=0)

        logger.info("Model info: %s", extra)
        return cls(model, preprocessor, tokenizer, decoder, device)

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes],
        audio_suffix: str = ".wav",
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        trim_silence: bool = True,
    ) -> Dict:
        """
        Transkripsi audio menjadi teks.

        Parameters
        ----------
        audio_input       : path ke file audio, atau bytes (upload Streamlit)
        audio_suffix      : ekstensi file jika input adalah bytes
        decoding_strategy : 'greedy' atau 'beam'
        beam_width        : digunakan jika decoding_strategy='beam'
        trim_silence      : potong keheningan sebelum inferensi

        Returns
        -------
        dict dengan kunci:
            text        : str   — hasil transkripsi
            confidence  : float — skor kepercayaan [0, 1]
            duration_sec: float — durasi audio (detik)
            latency_sec : float — waktu inferensi (detik)
            decoding    : str   — strategi decoding yang digunakan
            model_ready : bool  — False jika model belum dilatih
        """
        t_start = time.time()

        # --- Preprocessing ---
        try:
            if isinstance(audio_input, (str, Path)):
                features, info = self.preprocessor.process(
                    audio_input, trim=trim_silence, pad_truncate=True
                )
                duration = info["duration_processed_sec"]
            else:
                # bytes dari Streamlit
                features, info = self.preprocessor.process_waveform_bytes(
                    audio_input, suffix=audio_suffix,
                    trim=trim_silence, pad_truncate=True,
                )
                duration = info["duration_processed_sec"]
        except Exception as exc:
            logger.error("Preprocessing gagal: %s", exc)
            return self._error_result(str(exc))

        # --- Forward pass ---
        # preprocessor.process() returns [1, n_mels, T] (channel included)
        # model expects [B, n_mels, T], so squeeze channel then add batch dim
        features = features.squeeze(0).unsqueeze(0).to(self.device)  # [1, n_mels, T]

        try:
            with torch.no_grad():
                log_probs, output_lengths = self.model(features)
                # log_probs: [T', 1, vocab]
        except Exception as exc:
            logger.error("Forward pass gagal: %s", exc)
            return self._error_result(str(exc))

        # log_probs per sampel: [T', vocab]
        log_probs_single = log_probs[:, 0, :]
        out_len = output_lengths[0].item()

        # --- Decoding ---
        try:
            if decoding_strategy == "beam":
                text, confidence = self.decoder.beam_search_decode(
                    log_probs_single, out_len, beam_width=beam_width
                )
            else:
                text, confidence = self.decoder.greedy_decode(
                    log_probs_single, out_len
                )
        except Exception as exc:
            logger.error("Decoding gagal: %s", exc)
            return self._error_result(str(exc))

        latency = time.time() - t_start

        # Deteksi apakah model masih untrained (output sangat rendah confidence)
        model_ready = confidence > 0.01 and len(text.strip()) > 0

        result = {
            "text": text if text.strip() else "[Model belum dilatih — transkripsi tidak tersedia]",
            "confidence": round(confidence, 4),
            "duration_sec": round(duration, 3),
            "latency_sec": round(latency, 3),
            "decoding": decoding_strategy,
            "model_ready": model_ready,
            "feature_shape": info["feature_shape"],
        }

        logger.info(
            "Transkripsi selesai | durasi=%.2fs | latensi=%.3fs | "
            "confidence=%.3f | teks='%s'",
            duration, latency, confidence, text[:60],
        )
        return result

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
    ) -> List[Dict]:
        """
        Transkripsi batch file audio.

        Parameters
        ----------
        audio_paths : list path audio
        decoding_strategy, beam_width : sama dengan transcribe()

        Returns
        -------
        list hasil dict (satu per audio)
        """
        results = []
        for i, path in enumerate(audio_paths):
            logger.info("Batch %d/%d: %s", i + 1, len(audio_paths), path)
            result = self.transcribe(
                path,
                decoding_strategy=decoding_strategy,
                beam_width=beam_width,
            )
            result["file"] = str(path)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(error_msg: str) -> Dict:
        return {
            "text": f"[ERROR: {error_msg}]",
            "confidence": 0.0,
            "duration_sec": 0.0,
            "latency_sec": 0.0,
            "decoding": "none",
            "model_ready": False,
            "error": error_msg,
        }

    def is_model_available(self) -> bool:
        """
        Cek apakah model sudah pernah dilatih.
        Heuristik: cek apakah ada parameter non-zero di luar inisialisasi.
        """
        # Jika tokenizer punya lebih dari hanya special tokens, vocab sudah dibangun
        return self.tokenizer.vocab_size > len(CharacterTokenizer.SPECIAL_TOKENS)

    def get_model_info(self) -> Dict:
        """Kembalikan informasi ringkas model."""
        return {
            "architecture": "CNN-BiLSTM-CTC",
            "vocab_size": self.tokenizer.vocab_size,
            "n_mels": self.preprocessor.n_mels,
            "feature_type": self.preprocessor.feature_type,
            "max_duration_sec": self.preprocessor.max_duration_sec,
            "sample_rate": self.preprocessor.target_sample_rate,
            "device": str(self.device),
            "parameters": self.model.get_num_parameters(),
        }
