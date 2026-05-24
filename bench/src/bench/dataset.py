"""Dataset loader for the benchmark harness.

Resolves paths relative to `dataset/` at the repo root:
  - images/{anything}.{jpg,jpeg,png,heic}  — user-provided files; original names preserved
  - labels/{imageId}.json                  — canonical Receipt + _label block
  - splits/{train,val,test}.txt            — one imageId per line

Image files keep their original names (e.g. `IMG_1234.heic`); the labeler
computes a deterministic UUID per file (`imageId`) and the label JSON records
the source filename in its `_label.sourceFilename` field so the bench can
resolve image_id → file path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.schema import Receipt


# dataset.py lives at bench/src/bench/ — parents[3] is repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")


@dataclass
class LabeledSample:
    image_id: str
    image_path: Path
    label_path: Path
    label: Receipt
    label_status: str  # "verified" | "draft" | "rejected"
    source_filename: str | None


def iter_image_paths() -> list[Path]:
    """All images in dataset/images/ matching SUPPORTED_IMAGE_EXTS (case-insensitive)."""
    images_dir = DATASET_DIR / "images"
    if not images_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            out.append(p)
    return out


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


def find_image_path(image_id: str, source_filename: str | None = None) -> Path:
    """Resolve the on-disk image for `image_id`.

    Priority: explicit source_filename → {image_id}.{ext} lookup in images/.
    Raises FileNotFoundError if nothing matches.
    """
    images_dir = DATASET_DIR / "images"
    if source_filename:
        candidate = images_dir / source_filename
        if candidate.exists():
            return candidate
    for ext in SUPPORTED_IMAGE_EXTS:
        for variant in (ext, ext.upper()):
            candidate = images_dir / f"{image_id}{variant}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No image for {image_id} in {images_dir} "
        f"(tried sourceFilename={source_filename!r} and {SUPPORTED_IMAGE_EXTS})"
    )


def iter_labeled(statuses: set[str] = {"verified"}) -> list[str]:
    """Return imageIds of labels whose `_label.status` is in `statuses`.
    Useful when you haven't generated train/val/test splits yet.
    """
    labels_dir = DATASET_DIR / "labels"
    if not labels_dir.exists():
        return []
    out: list[str] = []
    for f in sorted(labels_dir.glob("*.json")):
        try:
            import json
            raw = json.loads(f.read_text())
            status = raw.get("_label", {}).get("status")
            if status in statuses:
                out.append(f.stem)
        except Exception:
            continue
    return out


def load_label(image_id: str) -> LabeledSample:
    label_path = DATASET_DIR / "labels" / f"{image_id}.json"
    if not label_path.exists():
        raise FileNotFoundError(f"No label file: {label_path}")
    raw = json.loads(label_path.read_text())
    label_block = raw.pop("_label", {})
    receipt = Receipt.model_validate(raw)

    source_filename = label_block.get("sourceFilename")
    image_path = find_image_path(image_id, source_filename=source_filename)

    return LabeledSample(
        image_id=image_id,
        image_path=image_path,
        label_path=label_path,
        label=receipt,
        label_status=label_block.get("status", "unknown"),
        source_filename=source_filename,
    )
