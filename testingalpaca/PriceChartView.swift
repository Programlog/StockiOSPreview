import SwiftUI
import Charts

struct PriceChartView: View {
    let bars: [Bar]
    let isLoading: Bool
    let errorMessage: String?

    @State private var selectedBar: Bar?
    @State private var isInteracting: Bool = false
    
    var body: some View {
        if #available(iOS 16.0, *) {
            Group {
                if isLoading {
                    ProgressView()
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
                        let chartColor: Color = (bars.first?.c ?? 0) <= (bars.last?.c ?? 0) ? .green : .pink
                        
                        VStack(spacing:0) {
                            ZStack {
                                if isInteracting, let bar = selectedBar {
                                    Text("\(bar.t, style: .date)")
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundColor(.primary)
                                        .transition(.opacity)
                                }
                            }
                            .frame(height: 20)         // Reserve consistent space for the date
                            .animation(.easeInOut(duration: 0.15), value: isInteracting)
                            ZStack(alignment: .top) {
                                Chart {
                                    ForEach(bars.indices, id: \.self) { i in
                                        let bar = bars[i]
                                        LineMark(
                                            x: .value("Date", i),    // Could be discrete or continuous
                                            y: .value("Price", bar.c)
                                        )
                                        .interpolationMethod(.linear)
                                        .foregroundStyle(chartColor)
                                        
                                        AreaMark(
                                            x: .value("Date", i),
                                            yStart: .value("Start", minClose),
                                            yEnd: .value("End", bar.c)
                                        )
                                        .foregroundStyle(
                                            LinearGradient(
                                                gradient: Gradient(colors: [
                                                    chartColor.opacity(0.3),
                                                    chartColor.opacity(0.03)
                                                ]),
                                                startPoint: .top,
                                                endPoint: .bottom
                                            )
                                        )
                                    }
                                }
                                .frame(height: 200)
                                .chartXAxis(.hidden)
                                .chartYScale(domain: minClose...maxClose)
                                .chartOverlay { proxy in
                                    GeometryReader { geometry in
                                        let plotAreaFrame = geometry[proxy.plotAreaFrame]
                                        
                                        Rectangle()
                                            .fill(Color.clear)
                                            .contentShape(Rectangle())
                                            .frame(width: plotAreaFrame.width, height: plotAreaFrame.height)
                                            .position(x: plotAreaFrame.midX, y: plotAreaFrame.midY)
                                            .gesture(
                                                DragGesture(minimumDistance: 0)
                                                    .onChanged { value in
                                                        isInteracting = true
                                                        let locationInPlot = CGPoint(
                                                            x: value.location.x - plotAreaFrame.origin.x,
                                                            y: value.location.y - plotAreaFrame.origin.y
                                                        )
                                                        // If we can map the X coordinate to an index, snap to the nearest data point
                                                        if let index: Double = proxy.value(atX: locationInPlot.x) {
                                                            let clampedIndex = Int(round(index))
                                                            if clampedIndex >= 0 && clampedIndex < bars.count {
                                                                selectedBar = bars[clampedIndex]
                                                            }
                                                        }
                                                    }
                                                    .onEnded { _ in
                                                        withAnimation(.easeInOut(duration:0.2)) {
                                                            isInteracting = false
                                                        }
                                                        
                                                    }
                                            )
                                        
                                        if isInteracting, let bar = selectedBar, let i = bars.firstIndex(where: { $0.t == bar.t }) {
                                            if let markerX = proxy.position(forX: Double(i)), let markerY = proxy.position(forY: bar.c)
                                            {
                                                // Draw vertical rule
                                                Path { path in
                                                    path.move(to: CGPoint(x: markerX, y: plotAreaFrame.minY))
                                                    path.addLine(to: CGPoint(x: markerX, y: plotAreaFrame.maxY))
                                                }
                                                .stroke(Color.secondary, style: StrokeStyle(lineWidth: 1, dash: [4]))
                                                
                                                // Ball marker
                                                Circle()
                                                    .fill(chartColor)
                                                    .frame(width: 12, height: 12)
                                                    .position(
                                                        x: markerX,
                                                        y: markerY
                                                    )
                                            }
                                        }
                                    }
                                }
                                
                                //
                                
                                if let bar = selectedBar {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("\(bar.t, style: .time)")
                                            .font(.caption)
                                        Text(String(format: "$%.2f", bar.c))
                                            .font(.caption.weight(.bold))
                                    }
                                    .padding(8)
                                    .background(.thinMaterial)
                                    .cornerRadius(9)
                                    .padding(.top, 32)
                                    .padding(.leading, 16)
                                }
                            }
                            .padding(.horizontal, 14)
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
