import os
from collections import defaultdict

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fer2013 import CLASS_NAMES, NAME_TO_ID

SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
TARGET_T = 300

RAVDESS_CODE = {
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def parse_label(filename: str) -> int | None:
    """
    Parse emotion label from RAVDESS filename.
    e.g. '03-01-05-01-01-01-01.wav' → label_id for 'angry'
    Returns None if the emotion code is not in our 7-class set.
    """
    parts = os.path.basename(filename).split("-")
    if len(parts) < 3:
        return None
    name = RAVDESS_CODE.get(parts[2])
    return NAME_TO_ID.get(name)


def scan_ravdess(audio_root: str) -> dict[int, list[str]]:
    """
    Scan RAVDESS audio_speech_actors_01-24/ directory.

    Returns:
        dict mapping label_id → list of .wav paths
    """
    by_label: dict = defaultdict(list)
    for actor in sorted(os.listdir(audio_root)):
        actor_dir = os.path.join(audio_root, actor)
        if not os.path.isdir(actor_dir):
            continue
        for f in os.listdir(actor_dir):
            if not f.lower().endswith(".wav"):
                continue
            lid = parse_label(f)
            if lid is not None:
                by_label[lid].append(os.path.join(actor_dir, f))
    return dict(by_label)


def wav_to_mel(
    path: str,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """
    Load a .wav file and convert to normalised log-Mel spectrogram.

    Returns:
        Tensor [1, n_mels, T]
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    y = librosa.util.fix_length(y, size=target_len)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256
    )
    db = librosa.power_to_db(mel, ref=np.max)
    db = (db - db.mean()) / (db.std() + 1e-9)
    return torch.tensor(db, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, T]


def pad_or_crop(mel: torch.Tensor, target_t: int = TARGET_T) -> torch.Tensor:
    """Pad or crop time axis to target_t frames."""
    t = mel.shape[2]
    if t < target_t:
        mel = F.pad(mel, (0, target_t - t))
    else:
        mel = mel[:, :, :target_t]
    return mel


class AudioOnlyDataset(Dataset):
    """
    Dataset for audio-only baseline.

    Args:
        audio_by_label: dict {label_id: [wav_path, ...]}
    """

    def __init__(self, audio_by_label: dict):
        self.samples = [
            (path, lid)
            for lid, paths in audio_by_label.items()
            for path in paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        mel = wav_to_mel(path)
        mel = pad_or_crop(mel, TARGET_T)
        return mel, label
