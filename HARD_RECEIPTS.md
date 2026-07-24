# Hard-to-parse receipts

## Current state (2026-07-24, after column-election + net-items batch)

1,183/1,183 extract, zero failures. **962 high (>= 0.85) - 166 medium
- 55 low.** Avg confidence 0.913. Day's arc this session: 877/228/78
-> 962/166/55, avg 0.885 -> 0.913.

Seven systematic fixes this session, each traced to a real receipt and
verified against the ground-truth photo, zero regressions across the
guard set:
1. Checksum + tax-marker COLUMN ELECTION — Safeway's two-column
   "Price | You Pay" layout now elects the paid column by matching a
   printed BALANCE/PAYMENT AMOUNT (± tax) to the cent, or by tax-marker
   density (IMG_6463 0.45 -> 0.97, IMG_5026 -> 1.00).
2. Two-letter tax markers ("FT" = WFM Food+Taxable) + net-of-savings
   re-anchoring (Whole Foods cluster IMG_8129/8253/6497/6580/6305 ->
   1.00; items are already net, discount was informational).
3. Drop informational discount LINE ITEMS (Old Navy net price + "Item
   Discount" breakdown; IMG_2107/7341 -> 1.00).
4. Impossible-tip guard (handwritten Clover tip "$7.00" OCR'd as "700";
   IMG_5130 -> 1.00).
5. Negative-tax drop + flipped-receipt totals-boundary guard.
6. Net-sales / total-taxes / taxable-amount non-item filter.
7. Unit+extended duplicate dedup (same value, one row, two X columns;
   Ace IMG_3658 0.30 -> 1.00).

### Remaining 55 low — the documented floor (37 under / 13 over / 5 zero)

**Genuine OCR under-extraction (37).** Confirmed by pulling the photos:
Vision does not emit price tokens for many items on crumpled / faded /
long thermal receipts. The 2x tiled re-scan already runs and cannot
recover text the sensor did not capture. Examples: Trader Joe's IMG_7345
(creases fragment 7 of 16 prices: "$11.46"->"46", "$9.23"->"$9"),
Grocery Outlet (5), Target long receipts (IMG_3060), Hashi, Sprouts.
Not fixable at the parse layer — needs better input.

**Diverse over-extraction (13), causes identified but each risky to fix:**
- Safeway ceiling-circle (IMG_6003): bogus FM total $3.99 makes the
  price-ceiling kill the $4.49 item; real total $16.11 prints twice.
- Safeway coupon-as-negative (IMG_4084): a real -$5.00 coupon is filtered
  as savings-metadata; keeping it (34.49 - 5 = 29.49) would reconcile,
  but that inverts the per-item "Member Savings" filtering we rely on
  elsewhere. Needs checksum-gated disambiguation — high regression risk.
- Safeway stacked regular/paid at same X (IMG_4084/7603): same value
  printed twice vertically (no discount, regular==paid); dedup requires
  a horizontal gap to avoid merging genuine same-priced stacked items.
- Restaurant qty (In-N-Out IMG_2035): "2 Hamburger" quantity misread.
- Oriental Grocery / Hashi two-column variants (IMG_6857/6642/7849):
  non-Safeway two-column layouts the election doesn't cover.
- Taco Mana IMG_1531 (2.9x): FM output triples items (malformed JSON).

**Zero-total (5), all correct or harmless:** IMG_6643 unreadable,
IMG_3557/8304 USPS/USPS $0 label receipts (correctly $0, excluded from
dollar charts), IMG_2562/6630 cut-off framing.

**Persistent one-offs:** TASSI IMG_5265 (items understate; total correct),
La Baguette IMG_2169 (x2), Mens Wearhouse IMG_1561 (return sign),
IMG_8245 Whole Foods 180-flip (descriptions mirror to the RIGHT of
prices — full orientation-mirroring is disproportionate risk for the
handful of flipped photos).

## Previous state (2026-07-23, after checksum-anchored-totals batch)

1,183/1,183 extract, zero failures (1,424 s wall, 6 workers).
**877 high (>= 0.85) - 228 medium - 78 low.** Avg confidence 0.885.
Dashboard: 1,179 receipts loaded, 1,103 in trends, 76 excluded by
the trust gate, trusted spend $48,753.83.

This round's fixes: checksum-anchored total re-selection (a totals-
label value matching the items sum to the cent beats FM's pick),
thousands-comma price shapes, department ring-up items, Old Navy
"Item Discount" metadata, dual-target checksum repair (subtotal and
total-tax-tip), Lytt struck-price drop. Low tail 98 -> 78.

Remaining 78 by cluster (worst first):
- **Safeway under-extraction (~20)** — the stacked metadata layout
  still loses rows when Vision fragments prices; largest cluster.
- **Grocery Outlet (~6), Whole Foods (~6), Trader Joe's (~4),
  Hashi (~3)** — same shape: items sum short of total.
- **Old Navy / Mens Wearhouse (4)** — return/discount sign flips;
  IMG_2107 improved (-27.99 -> +53.99) but still partial.
- **In-N-Out (2), DOHATSUTEN (2), La Baguette (2 dupes)** — layout
  one-offs.
- **TASSI IMG_5265** — items understate ($64.80 vs $2,224.80 total);
  total itself is correct.
- **True floor**: IMG_4412 faded Target, IMG_6643 unreadable,
  USPS $0-total label receipts (correct extractions, harmlessly
  excluded), IMG_3040 zero usable price text.


## Previous state (2026-07-22, after savings-total-circle round)

1,183/1,183 extract. **861 high (>= 0.85) - 224 medium - 98 low.**
Avg confidence 0.875. Dashboard: 1,083 of 1,179 receipts in trends,
96 excluded by the trust gate, trusted spend $48,139.47.

The savings-total vicious circle is fixed (bogus $1.00 totals no
longer nuke the item column via the price ceiling); cut-off photos
with self-consistent totals now score fairly. The remaining 98 are
the honest floor: faded thermal / crumpled paper where OCR produces
no usable price text, USPS/money-order formats, salon slips, and a
handful of deal-pricing one-offs. Day's arc: avg 0.73 -> 0.875,
high count 617 -> 861, low tail 343 -> 98.


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

## Residual after the tiled hi-res re-scan round (2026-07-22, latest)

The escalation path (conf < 0.6 → 3-band 2×-upscaled re-OCR → full
pipeline re-run → keep the higher score) rescued **121 of 296**
low-confidence receipts. Distribution now: **801 high (≥ 0.85)**,
207 medium, 175 low. Avg confidence 0.839.

Remaining 175 low-confidence receipts are the true hard core:
- under-half (55) + under (69) — text genuinely absent or illegible
  (thermal fade, crumples, fold shadows). 2× resolution didn't
  produce the missing observations either.
- over (27) + doubled (6) — mostly folded-receipt geometry where two
  physical column bands interleave (IMG_5085 class).
- close-low (12) — near-balanced sums with missing dates/merchants.

Next levers, roughly in value order: fold-aware column splitting for
the over/doubled class; EXIF-date fallback for undated receipts;
contrast enhancement (CLAHE-style) before OCR for thermal fade.

## Residual after the item-extraction round (2026-07-22, later)

Confidence distribution across all 1,183: **728 high (≥ 0.85)**,
159 medium, 296 low — up from 617/223/343 at the start of the day.
Remaining low-confidence buckets:

- **under-half (105) + under (106)** — items genuinely missing from
  extraction. Root causes seen: Vision drops price cents entirely
  ("$8" with no recoverable fragment — IMG_7496's chicken rows),
  faded thermal print, crumpled/folded geometry. Needs better OCR
  input (region re-scan at higher resolution?) more than parsing.
- **over (44) + doubled (18)** — over-extraction survivors; several
  are folded-receipt geometry (IMG_5085 photographs two column
  bands side by side).
- **close-low (16)** — sum within 15% but confidence dinged by
  missing dates/merchants or unresolved tax.
- The checksum repair (single-move drop/re-admit) fires only on
  exact ±1¢ residual matches; multi-item gaps stay for honest
  low-confidence flagging.

## Residual after the 2026-07-22 audit

All 1,183 receipts extract; 1,180 land in the dashboard (2 JPY
receipts + 1 undated excluded from dollar/time charts). Remaining
suspects worth a pass:

- ~~8 receipts with implausible tax~~ — FIXED by
  `reconcileTotalsArithmetic` (tax==total cleared on tax-exempt
  runs; subtotal==tax recovered as total−tax; BK's 9.375% rate
  recognized and decomposed; Philz remainder routed to TIP when
  the receipt prints a Tip label). All 124 receipts matching any
  suspect signature re-extracted; audit now reports zero
  implausible-tax and zero fractional-cent values dataset-wide.
  Note: several of the 8 still sit at conf 0.30 for a DIFFERENT
  reason — item extraction (IMG_5255 items doubled, IMG_5085
  over-extracted, IMG_2109 under-extracted). That's the next
  frontier, not a totals problem.
- **IMG_8142** — printed date unreadable (FM said "2026-03-00", OCR
  recovery found nothing valid). Honest sentinel; excluded from the
  time axis. Could fall back to photo EXIF date.
- **340 receipts below 0.6 confidence** — the honest hard-parse tail
  (faded thermal paper, crumpled, partial framing). Sorted worst-
  first in the dashboard's Recent Receipts panel.

## Fixed / no longer flagged (2026-07-22 full-dataset audit)

- **12 receipts with impossible dates** ("2026-09-42" from TJ's
  "03-15-2026 09:42" time-fusion; future dates) — calendar-valid date
  validation + OCR date recovery. All recovered real dates.
- **7 receipts stuck at the "1970-01-01" sentinel** (Ace, Hassett,
  Target, IKEA...) — `recoverDateFromOCR` found the printed date on
  every one. Sentinel also now scores as a missing date in confidence.
- **60 receipts with tax = taxable base** (Target's "CA TAX 9.375% on
  28.79" pattern) — tax candidates rejected when they echo the
  subtotal/items-sum or exceed 30% of total.
- **IMG_3389** ($8,854 "total" from a footer digit-run next to "TOTAL
  NUMBER OF ITEMS"; real $5.26) — item-count rows excluded from totals
  labels + whole-dollar-large re-anchoring.
- **IMG_7260 / IMG_7261** (¥5,160 Japan receipts summed as $5,160) —
  currency auto-detected as JPY; dashboard keeps non-USD out of dollar
  sums.
- **13 receipts failing with "Exceeded model context window"** — FM
  prompt now budgeted + retried with thinned rows on overflow. 12 of
  13 extract; IMG_1204 additionally needed the lexer-based truncation
  salvage (FM output cut mid-string at an escaped quote).

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
