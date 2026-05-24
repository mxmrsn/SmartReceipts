"""CLI entry point: `bench` (or `python -m bench.runner`)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from bench.dataset import (
    DATASET_DIR,
    iter_image_paths,
    iter_labeled,
    load_label,
    load_split,
)
from bench.pipelines.swift_pipeline import SwiftPipelineAdapter, list_pipelines
from bench.metrics import (
    score_sample,
    aggregate_field_scores,
    aggregate_line_items,
    teds,
    cer,
    LatencyStats,
)
from bench.report import (
    PipelineRunResult,
    SampleResult,
    write_summary_md,
    write_by_field_csv,
    write_html_diffs,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# ───────────────────── Inspection / smoke ─────────────────────


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
    """Smoke test: run vision-regex on one image and print the result."""
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
    images = iter_image_paths()
    by_ext: dict[str, int] = {}
    for p in images:
        by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1
    label_count = sum(1 for _ in (DATASET_DIR / "labels").glob("*.json"))
    verified = len(iter_labeled({"verified"}))
    drafts = len(iter_labeled({"draft"}))
    console.print(table)
    parts = [f"{ext}: {n}" for ext, n in sorted(by_ext.items())]
    detail = f"  ({', '.join(parts)})" if parts else ""
    console.print(f"[bold]images/[/bold] on disk: {len(images)}{detail}")
    console.print(f"[bold]labels/[/bold] on disk: {label_count}  ({verified} verified · {drafts} draft)")


@app.command()
def verify_label(image_id: str) -> None:
    """Verify a single label file parses against the canonical schema."""
    sample = load_label(image_id)
    console.print(f"[green]ok[/green] {image_id} status={sample.label_status} total={sample.label.totals.total}")


# ───────────────────── Full benchmark run ─────────────────────


@app.command()
def run(
    pipelines: str = typer.Option(
        "vision-fm,vision-regex",
        help="Comma-separated Swift pipeline ids to evaluate.",
    ),
    source: str = typer.Option(
        "verified",
        help="Which label set to evaluate against: verified | draft | both | test.",
    ),
    limit: Optional[int] = typer.Option(None, help="Cap samples for quick iteration."),
    output: Optional[Path] = typer.Option(
        None,
        help="Output directory. Defaults to bench/results/{timestamp}/.",
    ),
) -> None:
    """Run pipelines against a labeled subset and write summary + HTML diffs."""
    pipeline_ids = [p.strip() for p in pipelines.split(",") if p.strip()]
    label_ids = _collect_label_ids(source, limit)
    if not label_ids:
        console.print(f"[red]No labels found for source='{source}'.[/red]")
        raise typer.Exit(1)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = output or (Path(__file__).resolve().parents[3] / "bench" / "results" / timestamp)
    out.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Evaluating [bold]{len(pipeline_ids)}[/bold] pipeline(s) on "
        f"[bold]{len(label_ids)}[/bold] '{source}' label(s). Output → {out}"
    )

    # Load gold labels once
    samples = []
    for image_id in label_ids:
        try:
            samples.append(load_label(image_id))
        except FileNotFoundError as e:
            console.print(f"[yellow]skip[/yellow] {image_id}: {e}")
    if not samples:
        console.print("[red]No usable labels.[/red]")
        raise typer.Exit(1)

    # Run each pipeline
    results: list[PipelineRunResult] = []
    for pid in pipeline_ids:
        console.print(f"\n[bold]▶ {pid}[/bold]")
        results.append(_evaluate_pipeline(pid, samples))

    # Write reports
    write_summary_md(results, out / "summary.md")
    write_by_field_csv(results, out / "by_field.csv")
    write_html_diffs(results, out / "errors")

    # Console summary
    _print_console_summary(results)
    console.print(f"\n[green]✓[/green] Summary: {out / 'summary.md'}")


def _collect_label_ids(source: str, limit: Optional[int]) -> list[str]:
    if source == "test":
        ids = load_split("test")
    elif source == "verified":
        ids = iter_labeled({"verified"})
    elif source == "draft":
        ids = iter_labeled({"draft"})
    elif source == "both":
        ids = iter_labeled({"verified", "draft"})
    else:
        raise typer.BadParameter(f"Unknown source '{source}'")
    if limit:
        ids = ids[:limit]
    return ids


def _evaluate_pipeline(pipeline_id: str, samples) -> PipelineRunResult:
    adapter = SwiftPipelineAdapter(pipeline_id, display_name=pipeline_id)
    per_sample_results: list[SampleResult] = []
    per_sample_field_scores: list[dict] = []
    latencies: list[int] = []
    teds_scores: list[float] = []
    cer_scores: list[Optional[float]] = []
    errors = 0

    with console.status(f"  running {pipeline_id}…") as status:
        for i, sample in enumerate(samples, start=1):
            status.update(f"  running {pipeline_id} ({i}/{len(samples)}): {sample.image_path.name}")
            predicted = None
            raw_text = None
            err = None
            latency_ms = 0
            try:
                extraction = adapter.extract(sample.image_path)
                predicted = extraction.receipt
                raw_text = extraction.rawText
                latency_ms = extraction.latencyMs
                latencies.append(latency_ms)
            except Exception as e:
                err = str(e)
                errors += 1

            fs = score_sample(sample.label, predicted)
            per_sample_field_scores.append(fs)
            t = teds(sample.label, predicted)
            teds_scores.append(t)
            c = cer(None, raw_text)  # No gold rawText; CER skipped unless gold available
            cer_scores.append(c)

            per_sample_results.append(SampleResult(
                image_id=sample.image_id,
                image_path=sample.image_path,
                gold=sample.label,
                predicted=predicted,
                raw_text=raw_text,
                error=err,
                latency_ms=latency_ms,
                field_scores=fs,
                teds=t,
                cer=c,
            ))

    field_aggregate = aggregate_field_scores(per_sample_field_scores)
    li_aggregate = aggregate_line_items(per_sample_field_scores)

    return PipelineRunResult(
        pipeline_id=pipeline_id,
        sample_count=len(samples),
        error_count=errors,
        fields=field_aggregate,
        line_items=li_aggregate,
        teds_per_sample=teds_scores,
        cer_per_sample=cer_scores,
        latency=LatencyStats.from_samples(latencies),
        per_sample=per_sample_results,
    )


def _print_console_summary(results: list[PipelineRunResult]) -> None:
    t = Table(title="Pipeline scores")
    t.add_column("pipeline")
    t.add_column("samples", justify="right")
    t.add_column("err", justify="right")
    t.add_column("merch F1", justify="right")
    t.add_column("date F1", justify="right")
    t.add_column("total F1", justify="right")
    t.add_column("items F1", justify="right")
    t.add_column("TEDS", justify="right")
    t.add_column("p50 ms", justify="right")
    for r in results:
        def pct(v: float) -> str:
            return f"{v*100:.0f}%"
        merch = r.fields.get("merchant.name")
        date = r.fields.get("date.value")
        total = r.fields.get("totals.total")
        avg_teds = sum(r.teds_per_sample) / len(r.teds_per_sample) if r.teds_per_sample else 0
        t.add_row(
            r.pipeline_id,
            str(r.sample_count),
            str(r.error_count),
            pct(merch.f1) if merch else "—",
            pct(date.f1) if date else "—",
            pct(total.f1) if total else "—",
            pct(r.line_items.micro_f1),
            pct(avg_teds),
            str(r.latency.p50_ms),
        )
    console.print(t)


if __name__ == "__main__":
    app()
