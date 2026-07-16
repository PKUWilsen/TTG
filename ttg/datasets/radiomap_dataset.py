from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import re
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from ttg.utils.sparse_sampling import random_sample_1channel


def get_mat_numpy(path: str | Path) -> np.ndarray:
    """Load the first non-metadata variable from a MATLAB .mat file."""
    mat = loadmat(path)
    for key, value in mat.items():
        if key not in {"__header__", "__version__", "__globals__"}:
            return value
    raise ValueError(f"No data array found in {path}")


def get_sample_id(filename: str) -> Optional[int]:
    match = re.search(r"(\d+)\.mat$", filename)
    return int(match.group(1)) if match else None


def random_crop_single_channel(dense_image: np.ndarray, building_map: np.ndarray, crop_size: int):
    h, w = building_map.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Input map size {(h, w)} is smaller than crop size {crop_size}")
    row_offset = 0 if h == crop_size else np.random.randint(0, h - crop_size + 1)
    col_offset = 0 if w == crop_size else np.random.randint(0, w - crop_size + 1)
    dense_crop = dense_image[row_offset: row_offset + crop_size, col_offset: col_offset + crop_size]
    building_crop = building_map[row_offset: row_offset + crop_size, col_offset: col_offset + crop_size]
    return dense_crop, building_crop, row_offset, col_offset


class RadioMapDataset(Dataset):
    """Radio map dataset for the TTG coarse and refinement phases.

    Expected directory structure::

        data_root/
        ├── buildings_position/
        ├── receivedpower_5750MHz_mat/
        └── stations_position.txt

    The RSS map is normalized by ``(rss + rss_offset) / rss_scale`` before the
    final transform to [-1, 1]. Values greater than ``invalid_threshold`` are
    treated as invalid placeholders and set to ``invalid_fill_value``.
    """

    def __init__(
        self,
        data_root: str | Path,
        indices: Iterable[int],
        channel_num: int,
        sample_num: int,
        image_size: int = 256,
        max_stations: int = 3,
        rss_offset: float = 95.0,
        rss_scale: float = 66.0,
        invalid_threshold: float = -1.0,
        invalid_fill_value: float = -95.0,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.indices = list(indices)
        self.channel_num = int(channel_num)
        self.sample_num = int(sample_num)
        self.image_size = int(image_size)
        self.max_stations = int(max_stations)
        self.rss_offset = float(rss_offset)
        self.rss_scale = float(rss_scale)
        self.invalid_threshold = float(invalid_threshold)
        self.invalid_fill_value = float(invalid_fill_value)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        ])

        building_dir = self.data_root / "buildings_position"
        dense_dir = self.data_root / f"receivedpower_{self.channel_num}MHz_mat"
        if not building_dir.exists():
            raise FileNotFoundError(f"Missing building directory: {building_dir}")
        if not dense_dir.exists():
            raise FileNotFoundError(f"Missing RSS directory: {dense_dir}")

        self.dense_file_map = {
            sid: dense_dir / fname
            for fname in sorted(p.name for p in dense_dir.glob("*.mat"))
            if (sid := get_sample_id(fname)) is not None
        }
        self.building_file_map = {
            sid: building_dir / fname
            for fname in sorted(p.name for p in building_dir.glob("*.mat"))
            if (sid := get_sample_id(fname)) is not None
        }
        self.station_positions = self._load_station_positions()

    def _load_station_positions(self) -> Dict[int, list[int]]:
        station_file = self.data_root / "stations_position.txt"
        if not station_file.exists():
            return {}
        positions: Dict[int, list[int]] = {}
        with station_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    positions[i + 1] = [int(x) for x in line.split()]
        return positions

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_id = self.indices[idx]
        dense_path = self.dense_file_map.get(sample_id)
        building_path = self.building_file_map.get(sample_id)
        if dense_path is None or building_path is None:
            return None

        dense_image = get_mat_numpy(dense_path)
        building_map = get_mat_numpy(building_path)
        dense_image, building_map, row_offset, col_offset = random_crop_single_channel(
            dense_image, building_map, self.image_size
        )

        dense_image = dense_image.copy()
        dense_image[np.where(dense_image > self.invalid_threshold)] = self.invalid_fill_value
        dense_image = (dense_image + self.rss_offset) / self.rss_scale
        building_map = building_map / 255.0

        station_coords = self._crop_station_coords(sample_id, row_offset, col_offset)
        station_count = len(station_coords)
        station_coords_tensor = torch.full((self.max_stations, 2), -1.0, dtype=torch.float32)
        if station_count > 0:
            station_coords_tensor[:station_count, :] = torch.tensor(station_coords, dtype=torch.float32)

        sparse_image = random_sample_1channel(dense_image, self.sample_num)

        building_tensor = self.transform(building_map).to(torch.float32)
        sparse_tensor = self.transform(sparse_image).to(torch.float32)
        dense_tensor = self.transform(dense_image).to(torch.float32)
        return (
            sparse_tensor,
            dense_tensor,
            building_tensor,
            torch.tensor(station_count, dtype=torch.int),
            station_coords_tensor,
        )

    def _crop_station_coords(self, sample_id: int, row_offset: int, col_offset: int) -> list[list[float]]:
        original = self.station_positions.get(sample_id)
        if not original:
            return []
        coords = []
        pairs = [(original[i], original[i + 1]) for i in range(0, len(original), 2)]
        for y_orig, x_orig in pairs:
            x_new = x_orig - col_offset
            y_new = y_orig - row_offset
            if 0 <= x_new < self.image_size and 0 <= y_new < self.image_size:
                coords.append([float(x_new), float(y_new)])
        return coords


def build_dataset(config: Dict[str, Any], split: str, data_root: Optional[str | Path] = None) -> RadioMapDataset:
    data_cfg = config["data"]
    ranges = {
        "train": data_cfg.get("train_indices_range"),
        "val": data_cfg.get("val_indices_range"),
        "test": data_cfg.get("test_indices_range"),
        "gen": data_cfg.get("gen_indices_range"),
    }
    if split not in ranges or ranges[split] is None:
        raise ValueError(f"Unsupported or undefined split: {split}")
    start, end = ranges[split]
    root = Path(data_root or data_cfg["data_root"])
    return RadioMapDataset(
        data_root=root,
        indices=range(int(start), int(end)),
        channel_num=int(data_cfg["channel_num"]),
        sample_num=int(data_cfg["num_samples_sparse"]),
        image_size=int(data_cfg.get("img_height", 256)),
        max_stations=int(data_cfg.get("max_stations", 3)),
        rss_offset=float(data_cfg.get("rss_offset", 95.0)),
        rss_scale=float(data_cfg.get("rss_scale", 66.0)),
        invalid_threshold=float(data_cfg.get("invalid_threshold", -1.0)),
        invalid_fill_value=float(data_cfg.get("invalid_fill_value", -95.0)),
    )


def _collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        empty = torch.empty(0)
        return empty, empty, empty, empty, empty
    return torch.utils.data.dataloader.default_collate(batch)


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    batch_size: Optional[int] = None,
    data_root: Optional[str | Path] = None,
    distributed: bool = False,
    shuffle: Optional[bool] = None,
):
    dataset = build_dataset(config, split=split, data_root=data_root)
    data_cfg = config["data"]
    if batch_size is None:
        section = config.get("training", {}) if split in {"train", "val"} else config.get("evaluation", {})
        batch_size = int(section.get("batch_size", data_cfg.get("batch_size", 1)))
    num_workers = int(data_cfg.get("num_workers", 4))
    loader_args = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "num_workers": num_workers,
        "collate_fn": _collate_skip_none,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    if distributed:
        sampler = DistributedSampler(dataset)
        loader_args["sampler"] = sampler
        loader_args["shuffle"] = False
        return DataLoader(**loader_args), sampler
    loader_args["shuffle"] = bool(shuffle if shuffle is not None else split == "train")
    return DataLoader(**loader_args)
