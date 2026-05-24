"""Dataset loader for the benchmark harness.

Resolves paths relative to `dataset/` at the repo root:
  - images/{imageId}.jpg
  - labels/{imageId}.json   (canonical Receipt + _label block)
  - splits/{train,val,test}.txt  (one imageId per line)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.schema import Receipt


# dataset.py lives at bench/src/bench/ — parents[3] is repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"


@dataclass
class LabeledSample:
    image_id: str
    image_path: Path
    label_path: Path
    label: Receipt
    label_status: str  # "verified" | "draft" | "rejected"


def load_split(split: str) -> list[str]:
    """Return the list of imageIds for `train`, `val`, or `test`."""
    split_file = DATASET_DIR / "splits" / f"{split}.txt"
    if not split_file.exists():
        return []
    return [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_label(image_id: str) -> LabeledSample:
    label_path = DATASET_DIR / "labels" / f"{image_id}.json"
    if not label_path.exists():
        raise FileNotFoundError(f"No label file: {label_path}")
    raw = json.loads(label_path.read_text())
    label_block = raw.pop("_label", {})
    receipt = Receipt.model_validate(raw)

    image_jpg = DATASET_DIR / "images" / f"{image_id}.jpg"
    image_png = DATASET_DIR / "images" / f"{image_id}.png"
    image_path = image_jpg if image_jpg.exists() else image_png
    if not image_path.exists():
        raise FileNotFoundError(f"No image for {image_id} (looked for .jpg and .png)")

    return LabeledSample(
        image_id=image_id,
        image_path=image_path,
        label_path=label_path,
        label=receipt,
        label_status=label_block.get("status", "unknown"),
    )
