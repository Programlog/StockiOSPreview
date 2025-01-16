import SwiftUI
import Charts

struct PriceChartView: View {
    let bars: [Bar]
    let isLoading: Bool
    let errorMessage: String?

    @State private var selectedBar: Bar?
    
    var body: some View {
        if #available(iOS 16.0, *) {
            Group {
                if isLoading {
                    ProgressView("Loading...")
                        .frame(height: 200)
                } else if let errorMessage = errorMessage {
                    Text("Error: \(errorMessage)")
                        .foregroundColor(.red)
                        .frame(height: 200)
                } else if bars.isEmpty {
                    Text("No data available.")
                        .frame(height: 200)
                } else {
                    // 1) Compute min/max for dynamic y-axis
                    let closes = bars.map(\.c)
                    if let minClose = closes.min(), let maxClose = closes.max(), minClose < maxClose {
                        let lowerBound = minClose * 0.95
                        let upperBound = maxClose * 1.05

                        ZStack(alignment: .topLeading) {
                            Chart {
                                ForEach(bars.indices, id: \.self) { i in
                                    let bar = bars[i]
                                    LineMark(
                                        x: .value("Index", i),    // Could be discrete or continuous
                                        y: .value("Price", bar.c)
                                    )
                                    .interpolationMethod(.linear)
                                    .foregroundStyle(.blue)
                                }
                            }
                            .frame(height: 200)
                            .padding(.horizontal)
                            // 2) Hide X-axis labels
                            .chartXAxis(.hidden)
                            // 3) Dynamic y-axis domain
                            .chartYScale(domain: lowerBound...upperBound)
                            // Overlay for tooltip
                            .chartOverlay { proxy in
                                GeometryReader { geometry in
                                    Rectangle()
                                        .fill(Color.clear)
                                        .contentShape(Rectangle())
                                        .gesture(
                                            DragGesture(minimumDistance: 0)
                                                .onChanged { value in
                                                    // Convert x position to index
                                                    if let idx: Int = proxy.value(atX: value.location.x) {
                                                        // Snap to nearest valid index
                                                        if idx >= 0 && idx < bars.count {
                                                            selectedBar = bars[idx]
                                                        }
                                                    }
                                                }
                                        )
                                }
                            }
                            .shadow(color: .gray, radius: 5, x: 0, y: 6)

                            // Tooltip
                            if let bar = selectedBar {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("\(bar.t, style: .date) \(bar.t, style: .time)")
                                    Text(String(format: "$%.2f", bar.c))
                                }
                                .padding(6)
                                .background(.thinMaterial)
                                .cornerRadius(8)
                                .padding(.top, 10)
                                .padding(.leading, 10)
                            }
                        }
                    } else {
                        // Fallback if min/max can't be computed
                        Text("No valid price data.")
                            .frame(height: 200)
                    }
                }
            }
        } else {
            Text("Requires iOS 16+")
                .frame(height: 200)
        }
    }
}
