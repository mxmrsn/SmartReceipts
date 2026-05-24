"""CLI entry point: `bench` (or `python -m bench.runner`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bench.dataset import DATASET_DIR, load_label, load_split
from bench.pipelines.swift_pipeline import SwiftPipelineAdapter, list_pipelines

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def list_swift_pipelines() -> None:
    """List Swift OCR pipelines reachable via ocr-cli."""
    ids = list_pipelines()
    table = Table(title="Swift pipelines (via ocr-cli)")
    table.add_column("id")
    for pid in ids:
        table.add_row(pid)
    console.print(table)


@app.command()
def hello(image: Path = typer.Argument(..., help="Path to a single receipt image.")) -> None:
    """Smoke test: run the `vision-regex` pipeline on one image and print the result."""
    adapter = SwiftPipelineAdapter("vision-regex", "Apple Vision + Regex")
    result = adapter.extract(image)
    console.print_json(json.dumps(result.model_dump(mode="json")))


@app.command()
def stats() -> None:
    """Print the size of each dataset split."""
    table = Table(title="Dataset splits")
    table.add_column("split")
    table.add_column("count", justify="right")
    for split in ("train", "val", "test"):
        ids = load_split(split)
        table.add_row(split, str(len(ids)))
    image_count = sum(1 for _ in (DATASET_DIR / "images").glob("*.*") if _.suffix.lower() in (".jpg", ".png"))
    label_count = sum(1 for _ in (DATASET_DIR / "labels").glob("*.json"))
    console.print(table)
    console.print(f"[bold]images/[/bold] on disk: {image_count}")
    console.print(f"[bold]labels/[/bold] on disk: {label_count}")


@app.command()
def verify_label(image_id: str) -> None:
    """Verify a single label file parses against the canonical schema."""
    sample = load_label(image_id)
    console.print(f"[green]ok[/green] {image_id} status={sample.label_status} total={sample.label.totals.total}")


if __name__ == "__main__":
    app()
