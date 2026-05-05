import random
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .fer2013 import FER2013Dataset, default_transform
from .ravdess import wav_to_mel, pad_or_crop, TARGET_T


class PairedDataset(Dataset):
    """
    Label-aligned cross-dataset pairing.

    Each sample pairs:
        - a facial image from FER2013   (label = L)
        - a random speech clip from RAVDESS (same label L)

    No identity correspondence or temporal synchronisation required.

    Args:
        img_dataset    : FER2013Dataset instance
        audio_by_label : dict {label_id: [wav_path, ...]}
    """

    def __init__(self, img_dataset: FER2013Dataset, audio_by_label: dict):
        self.transform = img_dataset.transform
        self.audio_by_label = audio_by_label

        self.samples = [
            (p, lid)
            for p, lid in img_dataset.samples
            if audio_by_label.get(lid)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        audio_path = random.choice(self.audio_by_label[label])

        img = self.transform(Image.open(img_path).convert("RGB"))
        mel = pad_or_crop(wav_to_mel(audio_path), TARGET_T)

        return img, mel, label


def paired_collate(batch):
    """Custom collate — stacks (img, mel, label) tuples."""
    imgs, mels, labels = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(mels),
        torch.tensor(labels, dtype=torch.long),
    )


def split_audio(audio_by_label: dict, test_size: float = 0.2, seed: int = 42):
    """Split audio paths per class into train / val."""
    train, val = {}, {}
    for lid, paths in audio_by_label.items():
        if len(paths) >= 5:
            tr, te = train_test_split(paths, test_size=test_size, random_state=seed)
        else:
            tr, te = paths, []
        train[lid] = tr
        val[lid] = te
    return train, val


def make_sampler(dataset: PairedDataset) -> WeightedRandomSampler:
    """Weighted sampler for balanced class exposure."""
    labels = [lid for _, lid in dataset.samples]
    counts = Counter(labels)
    weights = torch.DoubleTensor([1.0 / counts[lid] for lid in labels])
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def build_loaders(
    img_train_ds: FER2013Dataset,
    img_val_ds: FER2013Dataset,
    audio_by_label: dict,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function: build train and val DataLoaders.

    Returns:
        train_loader, val_loader
    """
    audio_train, audio_val = split_audio(audio_by_label, seed=seed)

    train_ds = PairedDataset(img_train_ds, audio_train)
    val_ds = PairedDataset(img_val_ds, audio_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_sampler(train_ds),
        collate_fn=paired_collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=paired_collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
