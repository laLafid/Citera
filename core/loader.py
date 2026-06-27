"""
core/loader.py
Centralized image loading with randomization and one-by-one navigation.
"""

import os
import glob
import random
from pathlib import Path

VALID_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def load_images(
    folder: str = "dataset",
    n: int = None,
    seed: int = None,
    shuffle: bool = True,
) -> list[str]:
    """
    Load image paths from a folder.

    Args:
        folder:  Directory to scan.
        n:       Max images to return. None = all.
        seed:    Random seed for reproducibility.
        shuffle: Randomize order (default True).

    Returns:
        List of image file paths.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' tidak ditemukan.")

    paths: list[str] = []
    for ext in VALID_EXT:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))

    paths = list(dict.fromkeys(paths))  # deduplicate

    if not paths:
        raise ValueError(f"Tidak ada gambar ditemukan di folder '{folder}'.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    else:
        paths.sort()

    if n is not None:
        paths = paths[:n]

    return paths


def load_faces_or_dataset(n: int = None, seed: int = None) -> tuple[list[str], str]:
    """Load from 'faces' if exists, else 'dataset'. Returns (paths, folder_used)."""
    folder = "faces" if os.path.isdir("faces") else "dataset"
    return load_images(folder=folder, n=n, seed=seed), folder


class ImageNavigator:
    """
    One-image-at-a-time navigator with prev/next support.

    Usage:
        nav = ImageNavigator("dataset", seed=42)
        path = nav.current()   # get current image path
        path = nav.next()      # advance and return next path
        path = nav.prev()      # go back and return prev path
        nav.reshuffle()        # randomize order again
    """

    def __init__(self, folder: str = "dataset", seed: int = None):
        self.folder = folder
        self.seed   = seed
        self._paths: list[str] = []
        self._idx: int = 0
        self._load()

    def _load(self):
        self._paths = load_images(self.folder, shuffle=True, seed=self.seed)
        self._idx = 0

    # ── Navigation ───────────────────────────────────────────────

    def current(self) -> str | None:
        if not self._paths:
            return None
        return self._paths[self._idx]

    def next(self) -> str | None:
        if not self._paths:
            return None
        self._idx = (self._idx + 1) % len(self._paths)
        return self.current()

    def prev(self) -> str | None:
        if not self._paths:
            return None
        self._idx = (self._idx - 1) % len(self._paths)
        return self.current()

    def goto(self, idx: int) -> str | None:
        if not self._paths:
            return None
        self._idx = idx % len(self._paths)
        return self.current()

    def reshuffle(self, seed: int = None) -> str | None:
        """Re-randomize the list and return the new first image."""
        self.seed = seed
        self._load()
        return self.current()

    # ── Info ─────────────────────────────────────────────────────

    @property
    def index(self) -> int:
        return self._idx

    @property
    def total(self) -> int:
        return len(self._paths)

    @property
    def all_paths(self) -> list[str]:
        return list(self._paths)

    def has_images(self) -> bool:
        return bool(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __repr__(self) -> str:
        return (f"ImageNavigator(folder={self.folder!r}, "
                f"index={self._idx}/{len(self._paths)-1})")