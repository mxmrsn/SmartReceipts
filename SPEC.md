# Smart Receipts — App Specification & Implementation Plan

## Context

A SwiftUI + SwiftData Xcode template (the only existing code is the `Item` placeholder model and a list-view template) and ~1000 unannotated receipt images sitting on disk. The goal is to build a **fully on-device** iOS receipt-tracking app with a spend-trends dashboard, plus the supporting infrastructure to (a) **label** that dataset and (b) **benchmark** multiple on-device OCR pipelines against it before committing to a production path. The data set being unannotated is the critical bottleneck — without labels, you can't measure which pipeline is best, so the macOS labeling tool and benchmark harness are not "nice to haves" but the path to a defensible OCR choice.

Locked-in constraints:
- On-device only OCR (no cloud calls in production iOS)
- SwiftData local persistence (no iCloud, no backend)
- Full schema (header + line items + metadata)
- Spend-trends focused dashboard (Swift Charts)
- Custom SwiftUI macOS labeling tool (not Label Studio)
- Monorepo
- On-device Foundation Models for line-item categorization
- `VNDocumentCameraViewController` for capture

---

## Goals & Non-Goals

### Goals
1. iOS app that captures a receipt, OCRs + parses it on-device, stores it locally, and surfaces spend trends.
2. macOS labeling tool that converts ~1000 raw images into structured ground-truth JSON in the canonical schema.
3. Python benchmark harness that runs every candidate pipeline against the labeled set and emits per-field F1, TEDS, CER, latency, and memory metrics — so the iOS app's production pipeline is chosen empirically, not by feel.
4. A single source-of-truth schema shared across all three (Swift codegen + Python dataclasses).

### Non-Goals (Phase 1)
- Multi-device sync / CloudKit / accounts
- Sharing receipts / multi-user
- Tax export, business-vs-personal split (defer to Phase 2)
- Android, web, watchOS
- Cloud fallback in the iOS app (cloud APIs may appear in the benchmark harness only, as a reference ceiling)
- Receipt edit-after-save (Phase 1: review-before-save only)

---

## System Overview

Three artifacts, one schema:

```
                   ┌───────────────────────────┐
                   │  shared/schema/receipt.json│ ← single source of truth
                   └────────────┬──────────────┘
            ┌───────────────────┼──────────────────────┐
            ▼                   ▼                      ▼
   ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
   │  iOS app     │    │ macOS labeler  │    │  Python bench    │
   │ (Smart       │    │ (annotates     │    │  (scores         │
   │  Receipts)   │    │  1000 images)  │    │  pipelines)      │
   └──────┬───────┘    └────────┬───────┘    └────────▲─────────┘
          │                     │                     │
          │   reuses pipelines  │   exports labels    │
          └─── via Swift ───────┴─────────────────────┘
              Package (OCRKit)
```

`OCRKit` (Swift Package, in `packages/OCRKit/`) is the shared library that both the iOS app and the macOS labeler import. It exposes a uniform `OCRPipeline` protocol with multiple conforming implementations (Vision-only, Vision+FoundationModels, MLX-VLM, etc.). The Python bench reaches into the same pipelines via a thin Swift CLI wrapper (`tools/ocr-cli`) that wraps each iOS-shipped pipeline so on-device results are reproducible from Python.

---

## Canonical Receipt Schema

Single JSON file, `shared/schema/receipt.schema.json`, used to generate:
- `OCRKit.Receipt` (Swift, hand-mirrored for now; Sourcery codegen later)
- `bench/schema.py` (Python dataclass + Pydantic validator)
- Label-file format consumed by the labeler

Schema (slightly trimmed for readability):

```jsonc
{
  "imageId": "uuid",
  "header": {
    "merchant": { "name": "string", "address": "string?", "phone": "string?", "taxId": "string?" },
    "date":     { "value": "YYYY-MM-DD", "time": "HH:mm?" },
    "transactionId": "string?",
    "currency": "ISO 4217 (default USD)"
  },
  "lineItems": [
    {
      "description": "string",
      "quantity": "number?",
      "unitPrice": "number?",
      "totalPrice": "number",
      "category": "Food|Fuel|Groceries|Office|Transport|Lodging|Entertainment|Health|Other"
    }
  ],
  "totals": {
    "subtotal": "number?",
    "tax":      [{ "label": "string", "rate": "number?", "amount": "number" }],
    "discount": "number?",
    "tip":      "number?",
    "serviceCharge": "number?",
    "total":    "number"
  },
  "payment": {
    "method": "cash|card|check|other?",
    "cardLast4": "string?"
  },
  "provenance": {
    "pipelineId": "string",
    "modelVersion": "string",
    "confidence": "number",
    "fieldConfidence": { "merchant.name": 0.0, "totals.total": 0.0 },
    "bboxes": { "merchant.name": [0,0,0,0] }
  }
}
```

Field confidences and bboxes are mandatory because they drive (a) the iOS review screen's "needs your attention" highlighting and (b) active-learning sample selection in the labeler.

---

## iOS App Spec (`apps/ios/Smart Receipts`)

### Screens

| Screen | Purpose | Notes |
|---|---|---|
| **Capture** | Tap → `VNDocumentCameraViewController` sheet → returns multi-page scan | Auto-deskew, multi-page support |
| **Processing** | Brief progress view while OCRKit runs | Cancellable; shows pipeline name in debug builds |
| **Review** | Editable form pre-filled with extracted fields; low-confidence fields highlighted | Save commits to SwiftData |
| **Library** | Searchable/filterable list of saved receipts (by date, merchant, category, amount range) | Tap → detail |
| **Detail** | Full receipt view (image + parsed fields) | Edit, delete, re-OCR |
| **Dashboard** | Spend trends (see below) | Tab-bar primary |
| **Settings** | Pipeline selector (debug builds), data export, delete-all | Pipeline choice exposed in DEBUG only for now |

### SwiftData Model

```swift
@Model final class Receipt {
    @Attribute(.unique) var id: UUID
    var capturedAt: Date
    var imageRelativePath: String      // resolved against app's Documents/receipts/
    var receiptDate: Date?
    var merchantName: String?
    var total: Decimal?
    var currency: String               // ISO 4217
    var parsedPayloadJSON: Data        // full canonical Receipt JSON
    var pipelineId: String
    var overallConfidence: Double
    @Relationship(deleteRule: .cascade) var lineItems: [ReceiptLineItem]
}

@Model final class ReceiptLineItem {
    var description_: String
    var quantity: Decimal?
    var unitPrice: Decimal?
    var totalPrice: Decimal
    var category: String
    var receipt: Receipt?
}
```

Images live as JPEGs in `Documents/receipts/{uuid}.jpg`, not inside SwiftData — keeps the store small and migrations cheap. The full canonical JSON payload is also stored alongside the columnar fields so the model can evolve without losing detail.

### OCR Pipeline Architecture (`packages/OCRKit/`)

Single Swift Package, importable by both the iOS app and the macOS labeler.

```swift
public protocol OCRPipeline: Sendable {
    static var id: String { get }
    static var displayName: String { get }
    func extract(image: CGImage) async throws -> ExtractionResult
}

public struct ExtractionResult: Codable {
    public let receipt: Receipt
    public let latencyMs: Int
    public let peakMemoryMB: Int?
    public let rawText: String?
}
```

Conforming implementations (Phase 1 ships 1–3, Phase 2 adds the rest):

1. **`VisionOnlyPipeline`** — `VNRecognizeTextRequest(.accurate)` → regex/heuristic parser. The cheap baseline. Always available.
2. **`VisionPlusFoundationModelsPipeline`** *(production default)* — Vision for text extraction; Apple Foundation Models for schema-guided structured generation + line-item categorization. Requires iOS 18.x+ / FM-capable device.
3. **`MLXQwen25VLPipeline`** *(benchmark only)* — MLX Swift loading 4-bit Qwen2.5-VL-3B. End-to-end image → JSON. Heavy but interesting as a single-model reference.
4. **`MLXSmolVLMPipeline`** *(benchmark only)* — Lighter (~2.2B) for comparison on lower-end devices.

Adding a new pipeline = one file conforming to `OCRPipeline` + register in `OCRPipelineRegistry`. The benchmark harness picks them up automatically.

### Capture Flow

`VNDocumentCameraViewController` wrapped in `UIViewControllerRepresentable` — returns `[UIImage]` pre-deskewed/cropped. For multi-page receipts (long CVS-style), pages are concatenated vertically before OCR.

### Background Processing

Save flow is foreground (user is reviewing). Re-OCR on pipeline upgrade is queued via `BGProcessingTaskRequest` identifier `com.sciton.smartreceipts.reocr` so historical receipts can be re-parsed when a better pipeline ships, without blocking the UI.

### Dashboard (Swift Charts)

Phase 1 modules — all driven by `@Query` over SwiftData:
- **Monthly spend** — `BarMark` per calendar month, 12-month window default.
- **Category breakdown** — `SectorMark` (donut) for the active time window.
- **Top merchants** — Horizontal `BarMark`, top 10 by spend in window.
- **Trend line** — 7-day rolling average overlaid on daily bars.
- **Time-window picker** — Week / Month / Quarter / Year / All.

---

## macOS Labeling Tool (`apps/macos-labeler/`)

Native SwiftUI macOS app. Imports `OCRKit` directly so model-assisted pre-labeling uses the **same** pipelines the iOS app ships — closing the train/eval/serve loop tightly.

### Core Workflow

```
Import folder of images
         │
         ▼
Run all pipelines as "draft annotators" → store one draft per pipeline per image
         │
         ▼
Reviewer opens image → side-by-side panel:
   Left:  receipt image with overlaid bboxes (toggleable by pipeline)
   Right: editable canonical Receipt form, pre-filled from the chosen draft
         │
         ▼
Reviewer corrects → marks as "verified" → exports to `dataset/labels/{uuid}.json`
```

### Screens / Views

| View | Purpose |
|---|---|
| **Project browser** | List of label projects (folder + label-set pairs) |
| **Image grid** | Thumbnails, status pills (unlabeled / draft / verified / rejected), filter chips |
| **Labeling workspace** | Left: zoomable image + bbox overlay; Right: structured form; Bottom: keyboard shortcuts (`v` = verify, `n` = next, `[1-5]` = switch draft pipeline) |
| **Diff view** | Compare two pipelines' drafts on the same image |
| **Stats** | Per-pipeline draft accuracy vs. verified labels (live, as you review) |

### Label File Format

One file per image at `dataset/labels/{imageId}.json`, matching the canonical schema with an added `_label` block:

```jsonc
{
  "imageId": "...",
  "header": { /* canonical */ },
  "lineItems": [/* canonical */],
  "totals": { /* canonical */ },
  "_label": {
    "status": "verified|draft|rejected",
    "labeler": "max@sciton.com",
    "verifiedAt": "ISO8601",
    "sourcePipeline": "vision+fm",
    "notes": "string"
  }
}
```

### Active-Learning Loop

Stats view surfaces low-agreement images (pipelines disagree, or one pipeline's confidence is low) and prioritizes them in the queue — so reviewer time goes where it teaches the most.

### Why custom (vs. Label Studio)

3–4 weeks of build time buys: (a) Mac-native ergonomics, (b) **direct OCRKit reuse** so pre-labels come from the actual production code path, (c) bbox + structured-field labeling unified in one panel rather than two Label Studio templates stitched together, (d) export already in canonical format with no normalizer.

---

## Benchmark Harness (`bench/`)

Python (3.11+) project, separate from the Swift code.

### Layout

```
bench/
  schema.py                 # Pydantic mirror of canonical Receipt
  dataset.py                # Loads dataset/{images,labels}, train/val/test splits
  pipelines/
    base.py                 # PipelineAdapter ABC
    swift_pipeline.py       # Wraps `tools/ocr-cli` to run any Swift OCRPipeline
    cloud_textract.py       # Reference only — AWS Textract Expense
    cloud_mindee.py         # Reference only
  metrics/
    field_f1.py             # Per-field precision/recall/F1 with normalization
    teds.py                 # Tree edit distance over the canonical JSON tree
    cer.py                  # Char/word error rate over rawText (via jiwer)
    latency.py
  runner.py                 # Orchestrates: for each pipeline × test image → score
  report.py                 # Emits Markdown + CSV + per-image error gallery
```

### Splits

60/20/20 stratified by merchant-category. Fixed seed, splits checked into `dataset/splits/{train,val,test}.txt` so results are reproducible.

### Metrics (joint, not single-number)

1. **Field-level F1** — primary. Computed per field with normalization (currency → Decimal, date → ISO, merchant → casefold/strip). Aggregated per pipeline.
2. **TEDS** — tree edit distance over the canonical JSON tree; measures structural fidelity, the modern OCRBench-v2 / OmniDocBench standard.
3. **Exact-match** — strict, for headline reporting (% of receipts fully correct).
4. **CER on `rawText`** — sanity check that OCR text quality is reasonable independent of parsing.
5. **Latency p50 / p95** on a fixed device class.
6. **Peak memory (MB)** — only collectible from the Swift CLI; matters for older iPhones.

### Bridging Swift pipelines to Python

`tools/ocr-cli` is a tiny Swift command-line target that imports OCRKit and exposes:

```
ocr-cli --pipeline vision-regex --image path/to.jpg --json
```

Outputs canonical `Receipt` JSON to stdout. The Python `swift_pipeline.SwiftPipelineAdapter` shells out to it per image.

### Reports

`bench/report.py` writes:
- `bench/results/{run_id}/summary.md` — table of pipelines × metrics
- `bench/results/{run_id}/by_field.csv` — pivot by field × pipeline
- `bench/results/{run_id}/errors/{pipeline}/{imageId}.html` — visual diff (expected vs. predicted) for any image where any field mismatches

---

## Pipelines to Implement & Benchmark

| ID | Approach | Where it runs | Expected role |
|---|---|---|---|
| `vision-regex` | Apple Vision `.accurate` + regex parser | iOS / macOS | Floor — "how far can we get without an LLM" |
| `vision-fm` | Vision + Apple Foundation Models guided generation | iOS 18+ / macOS 15+ | **Likely production default** |
| `mlx-qwen25vl-3b-4bit` | MLX Swift end-to-end VLM | macOS M-series; iPhone 15+ marginal | Open-source single-model reference |
| `mlx-smolvlm-2.2b` | Lighter MLX VLM | Anywhere | Lower-tier device fallback candidate |
| `textract-expense` *(ref only)* | AWS Textract Expense | Cloud | Accuracy ceiling — never ships in app |
| `mindee-receipt` *(ref only)* | Mindee Receipt API | Cloud | Second ceiling reference |

---

## Monorepo Layout

```
Smart Receipts/
├── apps/
│   ├── ios/                            # Existing Xcode project moved here
│   │   └── Smart Receipts.xcodeproj
│   └── macos-labeler/                  # SwiftPM executable (Package.swift, M3)
│       ├── Package.swift
│       └── Sources/ReceiptLabeler/
├── packages/
│   └── OCRKit/                         # Swift Package
│       ├── Package.swift
│       └── Sources/OCRKit/
│           ├── Pipelines/
│           ├── Schema/
│           └── Registry.swift
├── tools/
│   └── ocr-cli/                        # SwiftPM executable, wraps OCRKit
├── bench/                              # Python benchmark harness
│   ├── pyproject.toml
│   └── ...
├── dataset/
│   ├── images/                         # Raw receipt images
│   ├── labels/                         # Per-image canonical JSON
│   └── splits/{train,val,test}.txt
├── shared/
│   └── schema/receipt.schema.json      # Single source of truth
└── SPEC.md
```

---

## Phasing / Milestones

**M1 — Foundations (Week 1)**
- Move existing iOS project into `apps/ios/`, create workspace
- Author `shared/schema/receipt.schema.json` + Swift `Receipt` + Python `Receipt`
- Stand up empty `OCRKit` package with `OCRPipeline` protocol + `VisionOnlyPipeline` (regex)
- Stand up empty `bench/` with `swift_pipeline.py` adapter + a single end-to-end "hello receipt" test

**M2 — iOS MVP (Weeks 2–3)**
- Capture screen with `VNDocumentCameraViewController`
- Review form bound to canonical schema
- SwiftData persistence + Library list
- Dashboard with Monthly Spend bar chart only
- Ships `VisionOnlyPipeline` as the working pipeline

**M3 — macOS Labeler (Weeks 3–5, overlaps M2)**
- Project browser, image grid, labeling workspace
- Hooks `OCRKit` for draft pre-labeling using `VisionOnlyPipeline`
- Export verified labels to `dataset/labels/`
- **Goal: label 100 receipts to unblock bench iteration**

**M4 — Foundation Models pipeline + bench buildout (Weeks 5–6)**
- Implement `VisionPlusFoundationModelsPipeline`
- Field-level F1, TEDS, CER metrics
- First full bench run: `vision-regex` vs. `vision-fm` on the 100-receipt labeled subset
- Add `errors/*.html` visual diff output

**M5 — Scale labeling + add MLX pipelines (Weeks 6–9)**
- Label remaining ~900 receipts using `vision-fm` as the draft annotator (faster review)
- Add `mlx-qwen25vl-3b-4bit` and `mlx-smolvlm-2.2b`
- Add cloud reference pipelines to the bench (Textract, Mindee)
- Full bench run; **pick production pipeline**

**M6 — Dashboard polish + categorization (Weeks 9–10)**
- Category donut, top merchants, trend line in Swift Charts
- On-device line-item categorization via Foundation Models guided generation
- BG re-OCR task for upgrading historical receipts

**M7 — Hardening (Week 10+)**
- Settings / data export / delete-all
- Empty states, error handling, low-confidence UX
- TestFlight build

---

## Verification

End-to-end checks per milestone:

1. **Schema cohesion (M1)** — round-trip a sample `Receipt` JSON through Swift and Python serializers; both produce byte-identical output for the canonical form.
2. **iOS capture → store (M2)** — run on device, scan a real receipt, confirm a `Receipt` row lands in SwiftData with the image at `Documents/receipts/{uuid}.jpg`. Inspect via `xcrun simctl get_app_container`.
3. **Labeler round-trip (M3)** — import a 5-image folder, draft → verify → export → re-import and confirm the verified labels survive serialization. Confirm `OCRKit` pre-labels are populated.
4. **Bench correctness (M4)** — hand-craft 3 known-good label files + 3 deliberately wrong predictions; confirm field-F1 and TEDS values match hand calculation.
5. **Bench scale (M5)** — full 200-image test split runs against every registered pipeline in under 30 min on M-series Mac. `summary.md` and `errors/` produced.
6. **Dashboard accuracy (M6)** — seed 50 known receipts spanning a year, confirm Swift Charts totals match a hand-summed spreadsheet within $0.01.
7. **On-device latency (M7)** — measure `vision-fm` p50/p95 on iPhone 15 (or your dev device). Target: < 4 s p50, < 8 s p95 for a single-page receipt.

---

## Open Questions to Revisit Mid-Build

- Whether `mlx-qwen25vl` is even viable on the minimum-target iPhone. Answer arrives at M5.
- Whether the Foundation Models token budget can hold a 30-line-item receipt in one shot, or whether the parse needs to be split (header pass + items pass). Answer arrives at M4 once we try it.
- Whether 1000 labeled receipts is enough to choose between similar pipelines, or if we need to augment with CORD/SROIE. Answer arrives at M5 when we look at confidence intervals on the F1 numbers.
- Category taxonomy — current 9-class list is a placeholder; revisit after seeing the real receipt distribution at M5.
