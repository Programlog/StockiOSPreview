import SwiftUI
import Charts

struct VolumeChartView: View {
    let bars: [Bar]

    var body: some View {
        if #available(iOS 16.0, *) {
            Chart {
                ForEach(bars.indices, id: \.self) { i in
                    let bar = bars[i]
                    LineMark(
                        x: .value("Time", bar.xLabel),
                        y: .value("Volume", bar.v)
                    )
                    .interpolationMethod(.linear)
                    .foregroundStyle(Color.green.opacity(0.6))
                }
            }
            .frame(height: 70)
            .padding(.horizontal)
            // Hide X-axis to simplify
            .chartXAxis(.hidden)
            // Also hide Y-axis for a minimal look
            .chartYAxis(.hidden)
        } else {
            Text("Requires iOS 16+")
                .frame(height: 70)
        }
    }
}

