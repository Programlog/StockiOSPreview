import SwiftUI

@MainActor
class StockDetailViewModel: ObservableObject {
    @Published var bars: [Bar] = []
    @Published var prediction: PredictionResponse?
    
    @Published var isLoading = false
    @Published var isLoadingPrediction = false
    @Published var errorMessage: String?

    private let service = StockDataService()

    func fetchData(symbol: String, chartRange: ChartRange) {
        Task {
            do {
                isLoading = true
                errorMessage = nil

                let start = chartRange.startDate()
                let end = ISO8601DateFormatter().string(from: Date())
                
                let fetchedBars = try await service.fetchStockData(
                    symbol: symbol,
                    timeframe: chartRange.alpacaTimeframe,
                    start: start,
                    end: end,
                    limit: 350 // Adjust as needed
                )
                self.bars = fetchedBars
            } catch {
                self.errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
    
    func fetchPrediction(symbol: String) {
        Task {
            do {
                isLoadingPrediction = true
                errorMessage = nil
                
                let prediction = try await service.fetchPrediction(symbol: symbol)
                self.prediction = prediction
            } catch {
                self.errorMessage = error.localizedDescription
            }
            isLoadingPrediction = false
        }
    }
}
