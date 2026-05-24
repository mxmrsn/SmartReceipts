"""Adapter that shells out to `tools/ocr-cli` to run any OCRKit pipeline.

This is the bridge that lets the Python harness measure on-device pipelines
without re-implementing them in Python. Every Swift pipeline registered in
`OCRPipelineRegistry` is reachable through this single adapter.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bench.pipelines.base import PipelineAdapter
from bench.schema import ExtractionResult


# swift_pipeline.py lives at bench/src/bench/pipelines/ — parents[4] is repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
OCR_CLI_DIR = REPO_ROOT / "tools" / "ocr-cli"


def _build_ocr_cli() -> Path:
    """Build the Swift CLI if needed; return the binary path."""
    subprocess.run(
        ["swift", "build", "-c", "release"],
        cwd=OCR_CLI_DIR,
        check=True,
        capture_output=True,
    )
    binary = OCR_CLI_DIR / ".build" / "release" / "ocr-cli"
    if not binary.exists():
        raise RuntimeError(f"ocr-cli binary missing at {binary} after build")
    return binary


def list_pipelines() -> list[str]:
    binary = _build_ocr_cli()
    out = subprocess.run(
        [str(binary), "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.stdout)
    return list(payload.get("pipelines", []))


class SwiftPipelineAdapter(PipelineAdapter):
    """Run a specific Swift pipeline via `ocr-cli`."""

    def __init__(self, pipeline_id: str, display_name: str | None = None) -> None:
        self.id = pipeline_id
        self.display_name = display_name or pipeline_id
        self._binary = _build_ocr_cli()

    def extract(self, image_path: Path) -> ExtractionResult:
        result = subprocess.run(
            [str(self._binary), "--pipeline", self.id, "--image", str(image_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ocr-cli failed for pipeline={self.id} image={image_path}:\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        # Apple frameworks (Vision, CoreML) occasionally print warnings like
        # "too few samples" to stdout. Slice from the first '{' so envelope
        # parsing is robust to that noise.
        payload = _extract_envelope(result.stdout, pipeline_id=self.id, image_path=image_path)
        if not payload.get("ok"):
            raise RuntimeError(f"ocr-cli reported failure: {payload.get('error')}")
        return ExtractionResult.model_validate(payload["result"])


def _extract_envelope(stdout: str, pipeline_id: str, image_path: Path) -> dict:
    """Locate and parse the JSON envelope in ocr-cli stdout.

    Tolerates leading framework-warning noise (e.g. Vision/CoreML printing
    "too few samples" to stdout) by slicing from the first '{' to the last
    '}'. Falls back to a clear error if no JSON object is found.
    """
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < 0 or end < start:
        snippet = stdout[:500] if stdout else "(empty stdout)"
        raise RuntimeError(
            f"ocr-cli for pipeline={pipeline_id} image={image_path} "
            f"produced no JSON envelope. stdout starts with: {snippet!r}"
        )
    body = stdout[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"ocr-cli envelope for pipeline={pipeline_id} image={image_path} "
            f"is not valid JSON ({e}). Body snippet: {body[:500]!r}"
        ) from e
