"""Pipeline adapter interface — all pipelines (Swift or cloud) implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bench.schema import ExtractionResult


class PipelineAdapter(ABC):
    """Abstract base for any OCR pipeline runnable from the harness."""

    id: str
    display_name: str

    @abstractmethod
    def extract(self, image_path: Path) -> ExtractionResult:
        """Run the pipeline on a single image and return the canonical result."""
        ...
