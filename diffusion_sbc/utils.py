from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def train_test_split_indices(n: int, test_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_ratio)))
    return idx[n_test:], idx[:n_test]


def grouped_train_test_split_indices(
    groups: np.ndarray,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split samples by group so no trajectory appears on both sides."""

    group_arr = np.asarray(groups).reshape(-1)
    unique = np.unique(group_arr)
    if unique.size < 2:
        return train_test_split_indices(group_arr.size, test_ratio, rng)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test_groups = max(1, min(unique.size - 1, int(round(unique.size * test_ratio))))
    test_groups = shuffled[:n_test_groups]
    test_mask = np.isin(group_arr, test_groups)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)
