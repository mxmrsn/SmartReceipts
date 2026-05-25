"""Batch pre-labeling: run a Swift OCR pipeline against every image in
`dataset/images/` and write a draft label file the macOS labeler can open
for review.

This is the M5 unblocker. Hand-running the labeler against ~1000 images
means waiting ~5s per image for FM to extract; batching here lets the
extraction happen overnight, leaving the user with ready-to-review drafts.

Safety:
  * never overwrites a `verified` label
  * by default never overwrites a `draft` either (use --overwrite-drafts)
  * any per-image failure is logged and we continue to the next image
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bench.dataset import DATASET_DIR, iter_image_paths
from bench.imageid import uuid_for_path
from bench.pipelines.swift_pipeline import SwiftPipelineAdapter

LABELS_DIR = DATASET_DIR / "labels"


@dataclass
class PrelabelOutcome:
    image_path: Path
    image_id: str
    status: str       # "wrote" | "skipped-existing" | "failed"
    reason: str = ""
    latency_ms: int = 0


def run_prelabel(
    pipeline_id: str,
    overwrite_drafts: bool = False,
    overwrite_all: bool = False,
    limit: Optional[int] = None,
    on_progress=None,
) -> list[PrelabelOutcome]:
    """Walk dataset/images/, run `pipeline_id` per image, write draft labels.

    Returns a list of per-image outcomes. `on_progress(i, total, outcome)` is
    called after each image — use it for live console output. The function
    itself does not print so it's testable.
    """
    images = iter_image_paths()
    if limit:
        images = images[:limit]

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    adapter = SwiftPipelineAdapter(pipeline_id, display_name=pipeline_id)

    outcomes: list[PrelabelOutcome] = []
    total = len(images)
    for i, image_path in enumerate(images, start=1):
        image_id = uuid_for_path(image_path)
        image_id_str = str(image_id).upper()
        label_path = LABELS_DIR / f"{image_id_str}.json"

        # Skip rules
        if label_path.exists():
            existing_status = _peek_status(label_path)
            if existing_status == "verified" and not overwrite_all:
                outcome = PrelabelOutcome(
                    image_path=image_path,
                    image_id=image_id_str,
                    status="skipped-existing",
                    reason=f"existing status='verified' (use --overwrite-all to replace)",
                )
                outcomes.append(outcome)
                if on_progress:
                    on_progress(i, total, outcome)
                continue
            if existing_status == "draft" and not (overwrite_drafts or overwrite_all):
                outcome = PrelabelOutcome(
                    image_path=image_path,
                    image_id=image_id_str,
                    status="skipped-existing",
                    reason="existing status='draft' (use --overwrite-drafts)",
                )
                outcomes.append(outcome)
                if on_progress:
                    on_progress(i, total, outcome)
                continue

        # Extract via Swift pipeline
        try:
            extraction = adapter.extract(image_path)
        except Exception as e:
            outcome = PrelabelOutcome(
                image_path=image_path,
                image_id=image_id_str,
                status="failed",
                reason=f"{type(e).__name__}: {e}",
            )
            outcomes.append(outcome)
            if on_progress:
                on_progress(i, total, outcome)
            continue

        # Write draft label file matching the labeler's on-disk shape.
        doc = _build_label_document(
            extraction_receipt=extraction.receipt,
            image_id=image_id_str,
            source_filename=image_path.name,
            source_pipeline=pipeline_id,
        )
        label_path.write_text(
            json.dumps(doc, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        outcome = PrelabelOutcome(
            image_path=image_path,
            image_id=image_id_str,
            status="wrote",
            latency_ms=extraction.latencyMs,
        )
        outcomes.append(outcome)
        if on_progress:
            on_progress(i, total, outcome)

    return outcomes


def _peek_status(label_path: Path) -> Optional[str]:
    """Read just the `_label.status` field without full schema validation —
    the file may pre-date a schema change."""
    try:
        data = json.loads(label_path.read_text())
        return data.get("_label", {}).get("status")
    except Exception:
        return None


def _build_label_document(
    *,
    extraction_receipt,  # Receipt Pydantic instance
    image_id: str,
    source_filename: str,
    source_pipeline: str,
) -> dict:
    """Round-trip the canonical receipt through JSON, swap in the
    deterministic imageId, then graft a `_label` block on top.
    """
    receipt_json = extraction_receipt.model_dump(mode="json")
    # The adapter's imageId is a random UUID assigned by the Swift pipeline;
    # overwrite with the labeler's deterministic id so the labeler picks
    # this draft up under the same key it would compute itself.
    receipt_json["imageId"] = image_id
    receipt_json["_label"] = {
        "status": "draft",
        "sourceFilename": source_filename,
        "sourcePipeline": source_pipeline,
    }
    return receipt_json
