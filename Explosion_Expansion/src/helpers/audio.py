import numpy as np
import torch
from typing import Tuple
import librosa

def extract_audio_array(path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract mono audio from video robustly.
    Works for very short clips without crashing MoviePy.
    """
    try:
        # Load audio directly via librosa (ffmpeg backend handles mp4)
        audio, orig_sr = librosa.load(path, sr=None, mono=True)
        if len(audio) == 0:
            raise ValueError("Empty audio stream")
        # Resample to target sr
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Warning: audio extraction failed for {path}, using silence. Error: {e}")
        return np.zeros(1, dtype=np.float32)


def log_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 64,
                        win_length: float = 0.025, hop_length: float = 0.010) -> torch.Tensor:
    """Compute log-mel spectrogram [1, Mels, T]."""
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S + 1e-10)
    logS = torch.from_numpy(logS).unsqueeze(0).float()   # [1,M,T]
    return logS
