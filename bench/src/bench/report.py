"""Bench reports: Markdown summary, CSV pivot, HTML per-image diff gallery."""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from bench.schema import Receipt
from bench.metrics import (
    FieldScore,
    LineItemsScore,
    LatencyStats,
)


@dataclass
class PipelineRunResult:
    pipeline_id: str
    sample_count: int
    error_count: int
    fields: dict[str, FieldScore]
    line_items: LineItemsScore
    teds_per_sample: list[float]
    cer_per_sample: list[Optional[float]]
    latency: LatencyStats
    per_sample: list["SampleResult"]


@dataclass
class SampleResult:
    image_id: str
    image_path: Path
    gold: Receipt
    predicted: Optional[Receipt]
    raw_text: Optional[str]
    error: Optional[str]
    latency_ms: int
    field_scores: dict
    teds: float
    cer: Optional[float]


# MARK: - Markdown summary

def write_summary_md(results: list[PipelineRunResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Bench summary")
    lines.append("")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    # Per-pipeline summary
    lines.append("| Pipeline | Samples | Errors | Merchant F1 | Date F1 | Total F1 | Items F1 (micro) | TEDS | CER | p50 ms | p95 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        merchant = r.fields.get("merchant.name")
        date = r.fields.get("date.value")
        total = r.fields.get("totals.total")
        teds = _mean(r.teds_per_sample)
        cer_vals = [v for v in r.cer_per_sample if v is not None]
        cer = _mean(cer_vals) if cer_vals else None
        lines.append(
            f"| `{r.pipeline_id}` | {r.sample_count} | {r.error_count} | "
            f"{_pct(merchant.f1 if merchant else 0)} | "
            f"{_pct(date.f1 if date else 0)} | "
            f"{_pct(total.f1 if total else 0)} | "
            f"{_pct(r.line_items.micro_f1)} | "
            f"{_pct(teds)} | "
            f"{('—' if cer is None else f'{cer*100:.1f}%')} | "
            f"{r.latency.p50_ms} | {r.latency.p95_ms} |"
        )
    lines.append("")

    # Per-field breakdown
    lines.append("## Per-field breakdown")
    for r in results:
        lines.append(f"### `{r.pipeline_id}`")
        lines.append("")
        lines.append("| Field | Correct | Gold | Pred | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key, s in r.fields.items():
            lines.append(
                f"| `{key}` | {s.correct} | {s.present_in_gold} | {s.present_in_pred} | "
                f"{_pct(s.precision)} | {_pct(s.recall)} | {_pct(s.f1)} |"
            )
        li = r.line_items
        lines.append(
            f"| `lineItems (micro)` | {li.micro_correct} | {li.micro_gold_count} | {li.micro_pred_count} | "
            f"{_pct(li.micro_precision)} | {_pct(li.micro_recall)} | {_pct(li.micro_f1)} |"
        )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


# MARK: - By-field CSV

def write_by_field_csv(results: list[PipelineRunResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pipeline_id", "field",
            "correct", "present_in_gold", "present_in_pred",
            "precision", "recall", "f1",
        ])
        for r in results:
            for key, s in r.fields.items():
                w.writerow([
                    r.pipeline_id, key,
                    s.correct, s.present_in_gold, s.present_in_pred,
                    f"{s.precision:.4f}", f"{s.recall:.4f}", f"{s.f1:.4f}",
                ])
            li = r.line_items
            w.writerow([
                r.pipeline_id, "lineItems (micro)",
                li.micro_correct, li.micro_gold_count, li.micro_pred_count,
                f"{li.micro_precision:.4f}", f"{li.micro_recall:.4f}", f"{li.micro_f1:.4f}",
            ])


# MARK: - HTML diff gallery

def write_html_diffs(results: list[PipelineRunResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        pipeline_dir = output_dir / r.pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        for sample in r.per_sample:
            if _is_perfect(sample):
                continue  # only generate pages for samples with diffs
            page = pipeline_dir / f"{sample.image_id}.html"
            page.write_text(_html_for_sample(r.pipeline_id, sample), encoding="utf-8")


def _is_perfect(sample: SampleResult) -> bool:
    if sample.error or sample.predicted is None:
        return False
    for entry in sample.field_scores.values():
        if isinstance(entry, dict) and "correct" in entry:
            if entry.get("present_in_gold") and not entry["correct"]:
                return False
    return True


def _html_for_sample(pipeline_id: str, sample: SampleResult) -> str:
    image_uri = sample.image_path.absolute().as_uri()
    rows: list[str] = []
    for field_key, entry in sample.field_scores.items():
        if field_key == "lineItems":
            continue
        gold = entry.get("gold_value", "")
        pred = entry.get("pred_value", "")
        match = entry.get("correct") and entry.get("present_in_gold")
        rows.append(
            f"<tr class='{'ok' if match else 'bad'}'>"
            f"<td>{html.escape(field_key)}</td>"
            f"<td>{html.escape(str(gold or '—'))}</td>"
            f"<td>{html.escape(str(pred or '—'))}</td>"
            f"</tr>"
        )
    li = sample.field_scores.get("lineItems", {})
    li_summary = (
        f"<p>Line items: {li.get('correct', 0)} matched of {li.get('gold_count', 0)} gold / "
        f"{li.get('pred_count', 0)} predicted (F1 {li.get('f1', 0):.0%})</p>"
        if li else ""
    )
    err_block = f"<p class='err'>Pipeline error: {html.escape(sample.error)}</p>" if sample.error else ""

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{pipeline_id} · {sample.image_id}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; }}
  .row {{ display: flex; gap: 20px; }}
  .img {{ flex: 1; }}
  .img img {{ max-width: 100%; max-height: 80vh; }}
  table {{ border-collapse: collapse; flex: 1; }}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; }}
  tr.ok td {{ background: #e6f7ea; }}
  tr.bad td {{ background: #fde7e7; }}
  .err {{ color: #b00; }}
  pre {{ background: #f4f4f4; padding: 10px; max-height: 200px; overflow: auto; }}
</style></head><body>
<h2>{html.escape(pipeline_id)} · {html.escape(sample.image_id)}</h2>
<p>Latency {sample.latency_ms} ms · TEDS {sample.teds:.0%} · CER {'—' if sample.cer is None else f"{sample.cer*100:.1f}%"}</p>
{err_block}
<div class='row'>
  <div class='img'><img src="{image_uri}" /></div>
  <div>
    <table>
      <tr><th>Field</th><th>Gold</th><th>Predicted</th></tr>
      {''.join(rows)}
    </table>
    {li_summary}
    <h3>Raw OCR text</h3>
    <pre>{html.escape(sample.raw_text or '(none)')}</pre>
  </div>
</div>
</body></html>
"""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(v: float) -> str:
    return f"{v*100:.1f}%"
