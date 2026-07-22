"""Build a fully self-contained dashboard.html from dataset/extracted/*.json.

Reads every per-receipt JSON, aggregates into a compact data structure, and
embeds it directly into dashboard.html. Result opens with a double-click —
no server, no external assets.

Usage:
    python3 dashboard/build.py

Output:
    dashboard/dashboard.html
"""

import datetime as _dt
import json, pathlib, re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "dataset/extracted"
TEMPLATE = ROOT / "dashboard/dashboard.template.html"
OUTPUT = ROOT / "dashboard/dashboard.html"


# ----- Sector categorization -----
# Merchant name (lowercased substring match) → sector. First hit wins.
SECTOR_RULES = [
    # Coffee shops
    ("philz",              "Coffee"),
    ("starbucks",          "Coffee"),
    ("blue bottle",        "Coffee"),
    ("peet",               "Coffee"),
    ("dunkin",             "Coffee"),
    # Groceries
    ("safeway",            "Groceries"),
    ("sprouts",            "Groceries"),
    ("whole foods",        "Groceries"),
    ("trader joe",         "Groceries"),
    ("grocery outlet",     "Groceries"),
    ("costco",             "Groceries"),
    ("oriental",           "Groceries"),
    ("hashi",              "Groceries"),
    ("kroger",             "Groceries"),
    ("albertsons",         "Groceries"),
    ("lucky",              "Groceries"),
    # Restaurants / food out
    ("chipotle",           "Restaurants"),
    ("mcdonald",           "Restaurants"),
    ("subway",             "Restaurants"),
    ("panera",             "Restaurants"),
    ("in-n-out",           "Restaurants"),
    ("in n out",           "Restaurants"),
    ("panda express",      "Restaurants"),
    ("burger",             "Restaurants"),
    ("pizza",              "Restaurants"),
    # Hardware / home
    ("ace hardware",       "Home & Hardware"),
    ("ace hard",           "Home & Hardware"),
    ("home depot",         "Home & Hardware"),
    ("lowe",               "Home & Hardware"),
    # Big box
    ("target",             "Big Box"),
    ("walmart",            "Big Box"),
    # Pet
    ("pet food",           "Pet"),
    ("petco",              "Pet"),
    ("petsmart",           "Pet"),
    # Fuel
    ("chevron",            "Fuel"),
    ("shell",              "Fuel"),
    ("mobil",              "Fuel"),
    ("arco",               "Fuel"),
    ("exxon",              "Fuel"),
    # Payments/services
    ("lytt",               "Services"),
]


def sector_for(merchant: str) -> str:
    if not merchant:
        return "Other"
    lo = merchant.lower()
    for needle, sector in SECTOR_RULES:
        if needle in lo:
            return sector
    return "Other"


def merchant_canonical(name: str) -> str:
    """Group receipts under one label per chain. Fuzzy — case-insensitive
    keyword match so "SAFEWAY C" and "Safeway" and "Safeway #304" collapse."""
    if not name:
        return "Unknown"
    lo = name.lower().strip()
    for needle, _ in SECTOR_RULES:
        if needle in lo:
            # Title-case the needle for a clean label.
            parts = needle.split()
            return " ".join(p.capitalize() for p in parts)
    # Fall back to the raw merchant name, trimmed.
    return name.strip()


def load_receipts():
    receipts = []
    skipped_bad_date = []
    skipped_non_usd = []
    for f in sorted(EXTRACTED.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not data.get("ok"):
            continue
        r = data["result"]["receipt"]
        h = r["header"]
        t = r["totals"]
        items = r["lineItems"]
        merchant_raw = (h.get("merchant") or {}).get("name") or ""
        date = (h.get("date") or {}).get("value") or ""
        # Skip obviously-broken date strings so charts don't get NaNs.
        if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
            continue
        # "1970-01-01" is the pipeline's no-date sentinel; and receipts
        # can't be from the future. Either way the date axis would lie.
        if date < "2015-01-01" or date > _dt.date.today().isoformat():
            skipped_bad_date.append(f.name)
            continue
        # Non-USD receipts (e.g. ¥ trip receipts) can't be summed into
        # dollar charts without conversion — keep them out and report.
        currency = (h.get("currency") or "USD").upper()
        if currency != "USD":
            skipped_non_usd.append(f.name)
            continue
        total = float(t.get("total") or 0)
        tax = float((t.get("tax") or [{}])[0].get("amount", 0)) if t.get("tax") else 0.0
        subtotal = float(t["subtotal"]) if t.get("subtotal") is not None else 0.0
        items_sum = sum(float(li["totalPrice"]) for li in items)
        conf = float(r["provenance"]["confidence"])

        receipts.append({
            "file": f.name.rsplit(".json", 1)[0],
            "date": date,
            "merchant_raw": merchant_raw,
            "merchant": merchant_canonical(merchant_raw),
            "sector": sector_for(merchant_raw),
            "total": total,
            "tax": tax,
            "subtotal": subtotal,
            "items_sum": items_sum,
            "item_count": len(items),
            "confidence": conf,
            "items": [
                {
                    "desc": li["description"],
                    "price": float(li["totalPrice"]),
                    "qty": float(li["quantity"]) if li.get("quantity") else None,
                }
                for li in items
            ],
        })
    if skipped_bad_date:
        print(f"Skipped {len(skipped_bad_date)} receipts with sentinel/implausible dates:")
        for n in skipped_bad_date:
            print(f"  {n}")
    if skipped_non_usd:
        print(f"Skipped {len(skipped_non_usd)} non-USD receipts:")
        for n in skipped_non_usd:
            print(f"  {n}")
    return receipts


def main():
    receipts = load_receipts()
    receipts.sort(key=lambda r: r["date"])
    print(f"Loaded {len(receipts)} receipts")

    if not TEMPLATE.exists():
        print(f"Missing template: {TEMPLATE}")
        return

    payload = json.dumps(receipts, separators=(",", ":"))
    template = TEMPLATE.read_text()
    html = template.replace("__DATA__", payload)
    OUTPUT.write_text(html)
    print(f"Wrote {OUTPUT}  ({len(html)/1024:.1f} KB, {len(receipts)} receipts)")


if __name__ == "__main__":
    main()
