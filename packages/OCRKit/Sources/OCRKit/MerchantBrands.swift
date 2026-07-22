import Foundation

/// Brand-name canonicalization for receipt merchants.
///
/// Receipts print the store name in many ways: all-caps, with store number,
/// with city ("TARGET T-1234 COLMA"), or via the OCR's lossy reading
/// ("HOME DEPOT" → "how doers"). The on-device LLM has no incentive to
/// prefer the canonical chain name, so we post-process by scanning the OCR
/// output for any known brand alias and snapping the merchant value to its
/// canonical form.
///
/// Adding a chain:
///   1. Add as many lowercase aliases as you expect to see in OCR output
///      (handle apostrophes both ways: "wendy's", "wendys").
///   2. Map each to the canonical display form.
///
/// We only scan the top ~30% of OCR lines so that incidental matches in
/// addresses or item names (e.g. an "Apple" snack on a grocery receipt)
/// don't accidentally rename the merchant. Brand names almost always sit
/// in the top header band of the printout.
public enum MerchantBrands {

    /// Ordered alias table. First match in OCR wins, so more-specific
    /// entries should appear above less-specific ones (e.g. "trader joe's"
    /// before "trader joe", though both map to the same canonical form).
    public static let aliases: [(needle: String, canonical: String)] = [
        // Big-box / membership
        ("target",                "Target"),
        ("walmart",               "Walmart"),
        ("wal-mart",              "Walmart"),
        ("wal mart",              "Walmart"),
        ("costco",                "Costco"),
        ("sam's club",            "Sam's Club"),
        ("sams club",             "Sam's Club"),
        ("bj's wholesale",        "BJ's Wholesale Club"),

        // Grocery
        ("whole foods",           "Whole Foods Market"),
        ("trader joe's",          "Trader Joe's"),
        ("trader joes",           "Trader Joe's"),
        ("sprouts farmers",       "Sprouts Farmers Market"),
        ("sprouts",               "Sprouts Farmers Market"),
        ("safeway",               "Safeway"),
        ("kroger",                "Kroger"),
        ("albertsons",            "Albertsons"),
        ("publix",                "Publix"),
        ("wegmans",               "Wegmans"),
        ("aldi",                  "Aldi"),
        ("lidl",                  "Lidl"),
        ("vons",                  "Vons"),
        ("ralphs",                "Ralphs"),
        ("h-e-b",                 "H-E-B"),
        ("heb ",                  "H-E-B"),
        ("food lion",             "Food Lion"),
        ("stop & shop",           "Stop & Shop"),
        ("harris teeter",         "Harris Teeter"),

        // Pharmacy / convenience
        ("walgreens",             "Walgreens"),
        ("cvs/pharmacy",          "CVS"),
        ("cvs pharmacy",          "CVS"),
        ("cvs",                   "CVS"),
        ("rite aid",              "Rite Aid"),
        ("7-eleven",              "7-Eleven"),
        ("7 eleven",              "7-Eleven"),
        ("seven-eleven",          "7-Eleven"),
        ("circle k",              "Circle K"),
        ("wawa",                  "Wawa"),
        ("sheetz",                "Sheetz"),

        // Coffee
        ("starbucks",             "Starbucks"),
        ("philz coffee",          "Philz Coffee"),
        ("peet's coffee",         "Peet's Coffee"),
        ("peets coffee",          "Peet's Coffee"),
        ("blue bottle",           "Blue Bottle Coffee"),
        ("dunkin",                "Dunkin'"),
        ("tim hortons",           "Tim Hortons"),

        // Fast food / restaurants
        ("mcdonald's",            "McDonald's"),
        ("mcdonalds",             "McDonald's"),
        ("burger king",           "Burger King"),
        ("wendy's",               "Wendy's"),
        ("wendys",                "Wendy's"),
        ("taco bell",             "Taco Bell"),
        ("chipotle",              "Chipotle Mexican Grill"),
        ("subway",                "Subway"),
        ("panera bread",          "Panera Bread"),
        ("panera",                "Panera Bread"),
        ("five guys",             "Five Guys"),
        ("in-n-out",              "In-N-Out Burger"),
        ("in n out",              "In-N-Out Burger"),
        ("shake shack",           "Shake Shack"),
        ("chick-fil-a",           "Chick-fil-A"),
        ("chick fil a",           "Chick-fil-A"),
        ("kfc",                   "KFC"),
        ("domino's",              "Domino's"),
        ("dominos",               "Domino's"),
        ("pizza hut",             "Pizza Hut"),
        ("papa john's",           "Papa John's"),
        ("papa johns",            "Papa John's"),

        // Home improvement / hardware
        ("home depot",            "Home Depot"),
        ("lowe's",                "Lowe's"),
        ("lowes",                 "Lowe's"),
        ("ace hardware",          "Ace Hardware"),
        ("menards",               "Menards"),

        // Gas
        ("chevron",               "Chevron"),
        ("shell oil",             "Shell"),
        ("exxon",                 "Exxon"),
        ("mobil",                 "Mobil"),
        ("arco",                  "ARCO"),
        ("valero",                "Valero"),
        ("speedway",              "Speedway"),

        // Office / electronics
        ("staples",               "Staples"),
        ("office depot",          "Office Depot"),
        ("officemax",             "OfficeMax"),
        ("best buy",              "Best Buy"),
        ("apple store",           "Apple Store"),
        ("microsoft store",       "Microsoft Store"),

        // Apparel / general retail
        ("ikea",                  "IKEA"),
        ("rei",                   "REI"),
        ("nordstrom",             "Nordstrom"),
        ("macy's",                "Macy's"),
        ("macys",                 "Macy's"),
        ("old navy",              "Old Navy"),
        ("tj maxx",               "TJ Maxx"),
        ("t.j. maxx",             "TJ Maxx"),
        ("marshalls",             "Marshalls"),
        ("ross dress for less",   "Ross Dress for Less"),

        // E-commerce
        ("amazon.com",            "Amazon"),
        ("amazon",                "Amazon"),
    ]

    /// Cities / suburbs that frequently appear at the top of receipts —
    /// the LLM sometimes picks them as the merchant when the actual store
    /// brand is elsewhere on the page. If FM's extracted merchant is one
    /// of these (case-insensitive) and we found a brand match in OCR,
    /// prefer the brand.
    public static let knownCities: Set<String> = [
        "colma", "anaheim", "san francisco", "san jose", "oakland",
        "berkeley", "palo alto", "mountain view", "sunnyvale",
        "cupertino", "santa clara", "san mateo", "redwood city",
        "san leandro", "fremont", "hayward", "milpitas",
        "los angeles", "long beach", "irvine", "pasadena",
        "santa monica", "burbank", "glendale", "torrance",
        "new york", "brooklyn", "queens", "manhattan", "bronx",
        "seattle", "bellevue", "redmond", "portland", "tacoma",
        "denver", "boulder", "austin", "houston", "dallas",
        "chicago", "boston", "cambridge", "miami", "atlanta",
    ]

    /// Look at the top of the OCR output (where the store header lives)
    /// and return the first matching canonical brand name, or nil if no
    /// known brand is present. `lines` should be the OCR observations
    /// in top-to-bottom reading order.
    public static func canonicalBrand(inTopOf lines: [String]) -> String? {
        guard !lines.isEmpty else { return nil }
        // Top third of the receipt; brand names live in the header band.
        // Cap at 20 lines so very long receipts don't pull in line-item
        // names that happen to contain a brand word.
        let topCount = min(20, max(1, lines.count / 3))
        let header = lines.prefix(topCount)
            .joined(separator: " ")
            .lowercased()

        // First alias to hit wins. Order in `aliases` matters: longer/more
        // specific aliases come earlier so e.g. "trader joe's" wins over
        // "trader" if both were in the table.
        //
        // We require needles to be surrounded by non-letter characters so
        // short aliases don't hit substrings of unrelated words: "mobil"
        // must not match "Mobile App", "ross" must not match "cross".
        for (needle, canonical) in aliases {
            if occursAsWord(needle, in: header) {
                return canonical
            }
        }
        return nil
    }

    /// True if `needle` appears in `haystack` bounded on both sides by
    /// non-letter characters (or string edges). Both strings must already
    /// be lowercased. Apostrophes inside the needle count as internal
    /// characters (so "wendy's" matches "wendy's cafe").
    private static func occursAsWord(_ needle: String, in haystack: String) -> Bool {
        guard !needle.isEmpty else { return false }
        var searchStart = haystack.startIndex
        while let match = haystack.range(of: needle, range: searchStart..<haystack.endIndex) {
            let beforeOK: Bool
            if match.lowerBound == haystack.startIndex {
                beforeOK = true
            } else {
                let prev = haystack[haystack.index(before: match.lowerBound)]
                beforeOK = !prev.isLetter
            }
            let afterOK: Bool
            if match.upperBound == haystack.endIndex {
                afterOK = true
            } else {
                let next = haystack[match.upperBound]
                afterOK = !next.isLetter
            }
            if beforeOK && afterOK { return true }
            searchStart = haystack.index(after: match.lowerBound)
        }
        return false
    }

    /// True if `merchant` looks like a bare city name and probably isn't
    /// the actual store. Used to decide whether to override FM's merchant
    /// output with a brand we found in the OCR text.
    public static func looksLikeCity(_ merchant: String) -> Bool {
        let normalized = merchant
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return knownCities.contains(normalized)
    }
}
