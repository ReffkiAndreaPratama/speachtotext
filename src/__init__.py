"""
Speech-to-Text Bahasa Bengkulu
================================
Package src: modul-modul utama untuk pipeline speech recognition.

Modul:
    - preprocessing : konversi dan ekstraksi fitur audio
    - dataset       : loader dan manajemen dataset
    - model         : arsitektur deep learning
    - trainer       : loop training dan checkpointing
    - inference     : engine inferensi
"""

from src.preprocessing import AudioPreprocessor
from src.dataset import BengkuluSpeechDataset, DatasetManager
from src.model import SpeechRecognitionModel, ModelConfig
from src.trainer import Trainer, TrainingConfig
from src.inference import InferenceEngine

__version__ = "1.0.0"
__author__ = "Speech-to-Text Bahasa Bengkulu Project"

__all__ = [
    "AudioPreprocessor",
    "BengkuluSpeechDataset",
    "DatasetManager",
    "SpeechRecognitionModel",
    "ModelConfig",
    "Trainer",
    "TrainingConfig",
    "InferenceEngine",
]
