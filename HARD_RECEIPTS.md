# Hard-to-parse receipts

Living TODO list of receipts that don't yet extract cleanly. Cross out
(`~~IMG_XXXX~~`) as they're resolved. When a whole failure mode is
retired, remove the section.

Confidence bands referenced below map to `receipt.provenance.confidence`:
- **high** ≥ 0.85 (dashboard shows green)
- **medium** 0.60–0.85 (yellow)
- **low** < 0.60 (red — worth investigating)

## In active development

### Safeway — long two-column layout (metadata + paid)

Safeway prints every item as three stacked rows:
```
  BROCCOLI CROWNS           ← description
  Regular Price   4.99      ← metadata, indented column
  Member Savings  1.00-     ← metadata, indented column
                  3.99 S    ← paid price, rightmost column, tax marker
```

Fixed in this iteration by (a) fuzzy-matching typo variants like
`Reguler Price`, `Menber Savings`, `Member Sovings` in the metadata
filter and (b) preferring the rightmost cluster when price observations
are bimodal in X. Verified working on IMG_1788, IMG_5046, IMG_8571.

Still flagged (`conf < 0.85`):

- IMG_1697 (conf 0.37, diff $22.27) — 7/? items, missing large items in the middle
- IMG_2031 (conf 0.45, diff $24.56) — under-extracting, sum $37 vs $62 total
- IMG_2083 (conf 0.45, diff $14.30) — 4 items, missing ~$14 of merchandise
- IMG_1425 (conf 0.45, diff $5.99) — 4 items, missing one
- IMG_1815 (conf 0.45, diff -$2.89) — over-extracting slightly
- IMG_1479 (conf 0.45, diff $4.20) — under-extraction
- IMG_2460 (conf 0.67, diff $4.68) — 20 items, close but not exact
- IMG_1255 (conf 0.30, diff $13.31) — 1 item only, extraction failed most rows
- IMG_1430 (conf 0.10) — SUM MATCHES ($9.98) but tax also $9.98 (bogus)
- IMG_1953 (conf 0.45, diff $3.74) — 1 item on a $9.73 receipt
- IMG_2152 (conf 0.45, diff $0.20) — 4 items, very close, off by rounding

### In-N-Out Burger — over-extracting the total line

FM treats "DRIVE-Take Out $14.15" as a line item, but $14.15 is actually
the subtotal / total value being repeated in a "TAKE OUT" summary row.

- IMG_1250 (conf 0.10, sum $28.10 vs total $16.21)
- IMG_1564 (conf 0.10)
- IMG_1844 (conf 0.30)
- IMG_2035 (conf 0.45)

### Target — total field grabbing wrong value

Two extracted items summing to way less than total; FM likely picked
"loyalty savings" or a subtotal as the total.

- IMG_1203 (conf 0.30, total $513.37 vs items sum $55.89) — clearly the
  wrong "total" field; needs cross-check against receipt max value
- IMG_1272 (conf 0.30, total $142.11 vs items $69.93)
- IMG_1429 (conf 0.30, total $9.98 vs items $4.49)
- IMG_1251 (conf 0.45, 18 items sum $69 vs total $100)

### Old Navy / Menswear — negative sum

Return receipts where the SUM is negative but the total is a positive
refund amount. Signs are getting cross-wired.

- IMG_1561 THE MENS WEARHOUSE (items sum -$174.28 vs total $131.23)
- IMG_2107 OLD NAVY (items sum -$27.99 vs total $89.67)

### La Baguette / ULTA / BK — huge under-extraction

Only 1 item extracted from a receipt with several. Layout unknown yet.

- IMG_2169 La Baguette (conf 0.30, 1 item $1.04 vs total $13.00)
- IMG_2108 ULTA (conf 0.45)
- IMG_2109 BK (conf 0.45)

### Sprouts — variable

- IMG_2105 (conf 0.45, sum $20.13 vs total $27.99) — long receipt
- IMG_8132 (conf ?) — 2 items on a $21.25 receipt (huge under-extraction)

### Home Depot / Walgreens / Madewell — miscellaneous

- IMG_2141 Home Depot (conf 0.10, sum $17.98 vs total $19.60) — off-by-tax
- IMG_1741 Walgreens (conf 0.10)
- IMG_2171 Madewell (conf 0.10)

## Fixed / no longer flagged

- IMG_5785 (Grocery Outlet on wooden table, rotated 90°) — now extracts
  10 items summing to $30.38 exactly. Fixed by
  `VNDetectDocumentSegmentation` + `CIPerspectiveCorrection`
  preprocessing for small/rotated receipts.
- IMG_5046 (Safeway with metadata columns) — now perfect ($53.88 exact).
  Fixed by rightmost-cluster column detection when X observations are
  bimodal (metadata column at ~0.58, real prices at ~0.66).
- IMG_1788 (Safeway with `Reguler Price` / `Menber Savings` / `Member
  Sovings` typo variants) — was 3 metadata items summing $1.51 against
  $31.77 total; now 7 real items summing $28.28 (only $3.49 short — a
  price-fragment case where Vision split "3.49" into "3" and "49 S").
  Fixed by Levenshtein-based fuzzy matching in the metadata filter
  and the column-anchored labelYs precompute.
- IMG_2460 (Safeway with GOYA/ADOBO fragmented prices) — dropped from
  $173 items-sum bogus down to $70.81 (still $4.68 short but reasonable).
  Fixed by split-price fragment merge.
- IMG_5190, IMG_5785 misidentified as "Mobil" — merchant matcher now
  requires word boundaries so "Mobil" no longer matches "Mobile App".
- IMG_1250 (In-N-Out DRIVE-Take Out subtotal treated as $14.15 item) —
  sum dropped from $28.10 to $13.95 vs $16.21 total. The remaining $2.26
  gap is a distinct issue (FM misreads "2 Hamburger" quantity). Fixed by
  adding restaurant order-type phrases to totals-boundary + non-item
  filters.

## Investigation notes

- The Foundation Models call is deterministic (greedy, temp 0), so
  re-running any receipt reproduces the same output. Use
  `OCR_DEBUG_PROMPT=1` and `OCR_DEBUG_FM=1` env vars on `ocr-cli` to
  see the exact FM prompt and raw response.
- `OCR_DEBUG_COL=1` prints the column-anchored items before they
  become the final line-item list.
- `OCR_DEBUG_META=1` prints which observations get dropped by the
  metadata filter and why.
- The confidence score's dominant signal is "items sum to total ± tax",
  so a low-confidence receipt almost always has an items-mismatch.
- After changing pipeline code, delete the affected receipts from
  `dataset/extracted/*.json` and re-run `scratchpad/extract_all.py` to
  refresh them; the extractor caches on disk to survive interruptions.
