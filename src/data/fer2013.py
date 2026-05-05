import os
import random
from glob import glob
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

IMG_SIZE = 224


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def scan_folder(root: str) -> list[tuple[str, int]]:
    """
    Scan FER2013 train/ or test/ directory.

    Expected layout:
        root/
            angry/  *.jpg
            happy/  *.jpg
            ...

    Returns:
        list of (image_path, label_id)
    """
    samples = []
    for cls in os.listdir(root):
        if cls not in NAME_TO_ID:
            continue
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        lid = NAME_TO_ID[cls]
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob(os.path.join(cls_dir, ext)):
                samples.append((p, lid))
    return samples


def cap_per_class(samples: list, max_per_class: int, seed: int = 42) -> list:
    """Randomly downsample each class to max_per_class samples."""
    rng = random.Random(seed)
    by_label: dict = defaultdict(list)
    for p, lid in samples:
        by_label[lid].append(p)

    capped = []
    for lid, paths in by_label.items():
        if len(paths) > max_per_class:
            paths = rng.sample(paths, max_per_class)
        capped.extend([(p, lid) for p in paths])
    return capped


class FER2013Dataset(Dataset):
    """
    FER2013 image dataset.

    Args:
        root         : path to train/ or test/ folder
        transform    : torchvision transforms (default: ImageNet normalisation)
        max_per_class: cap samples per class for balanced training (None = no cap)
        seed         : random seed for sampling
    """

    def __init__(
        self,
        root: str,
        transform=None,
        max_per_class: int | None = None,
        seed: int = 42,
    ):
        samples = scan_folder(root)
        if max_per_class is not None:
            samples = cap_per_class(samples, max_per_class, seed=seed)
        self.samples = samples
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
