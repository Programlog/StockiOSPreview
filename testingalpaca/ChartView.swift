import SwiftUI

struct ChartView: View {
    let bars: [Bar]
    let isLoading: Bool
    let errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            PriceChartView(
                bars: bars,
                isLoading: isLoading,
                errorMessage: errorMessage
            )
            .frame(height: 200)

            Divider()
                .frame(height: 2)

            VolumeChartView(bars: bars)
                .frame(height: 70)
            Divider()
        }
    }
}
