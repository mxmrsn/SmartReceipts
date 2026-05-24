import Charts
import SwiftData
import SwiftUI

/// Phase-1 dashboard: monthly spend over the last 12 months as a BarMark.
/// More charts (category donut, top merchants, trend line) arrive in M6.
struct DashboardView: View {

    @Query private var receipts: [Receipt]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    summary
                    chartCard
                }
                .padding()
            }
            .navigationTitle("Dashboard")
        }
    }

    // MARK: - Pieces

    private var summary: some View {
        let buckets = monthlyTotals()
        let totalSpend = buckets.map(\.amount).reduce(Decimal(0), +)
        let monthsCovered = buckets.filter { $0.amount > 0 }.count

        return VStack(alignment: .leading, spacing: 4) {
            Text("Last 12 months")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(totalSpend, format: .currency(code: primaryCurrency))
                    .font(.largeTitle.weight(.semibold))
                    .monospacedDigit()
                Text("across \(monthsCovered) month\(monthsCovered == 1 ? "" : "s")")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var chartCard: some View {
        let buckets = monthlyTotals()
        return VStack(alignment: .leading, spacing: 12) {
            Text("Monthly spend")
                .font(.headline)
            if receipts.isEmpty {
                EmptyChart()
            } else {
                Chart(buckets) { bucket in
                    BarMark(
                        x: .value("Month", bucket.month, unit: .month),
                        y: .value("Total", NSDecimalNumber(decimal: bucket.amount).doubleValue)
                    )
                    .foregroundStyle(.tint)
                    .cornerRadius(4)
                }
                .frame(height: 220)
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Decimal(v), format: .currency(code: primaryCurrency).precision(.fractionLength(0)))
                            }
                        }
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .stride(by: .month)) { value in
                        AxisGridLine()
                        AxisValueLabel(format: .dateTime.month(.narrow), centered: true)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.background.secondary)
        )
    }

    // MARK: - Aggregation

    /// 12 contiguous month buckets ending in the current month, each summed.
    private func monthlyTotals() -> [MonthBucket] {
        let calendar = Calendar.current
        let now = Date()
        guard let endOfThisMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: now))
        else { return [] }

        let months: [Date] = (0..<12).reversed().compactMap { offset in
            calendar.date(byAdding: .month, value: -offset, to: endOfThisMonth)
        }

        let monthKeys = months.map { Self.monthKey(for: $0, calendar: calendar) }
        var sums: [String: Decimal] = Dictionary(uniqueKeysWithValues: monthKeys.map { ($0, Decimal(0)) })

        for receipt in receipts {
            guard let total = receipt.total else { continue }
            let date = receipt.receiptDate ?? receipt.capturedAt
            let key = Self.monthKey(for: date, calendar: calendar)
            if sums[key] != nil {
                sums[key]! += total
            }
        }

        return zip(months, monthKeys).map { date, key in
            MonthBucket(month: date, amount: sums[key] ?? 0)
        }
    }

    private static func monthKey(for date: Date, calendar: Calendar) -> String {
        let comps = calendar.dateComponents([.year, .month], from: date)
        return "\(comps.year ?? 0)-\(comps.month ?? 0)"
    }

    /// The currency used by the most recent receipt, defaulting to USD.
    private var primaryCurrency: String {
        receipts.first?.currency ?? "USD"
    }
}

// MARK: - Bucket

private struct MonthBucket: Identifiable {
    var id: Date { month }
    let month: Date
    let amount: Decimal
}

// MARK: - Empty state

private struct EmptyChart: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(.secondary)
            Text("No data yet")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 220)
    }
}
