//
//  Stock.swift
//  testingalpaca
//
//  Created by Varun Kota on 1/11/25.
//


// HomeViewModel.swift
import SwiftUI

struct Stock: Identifiable {
    var id: String { symbol }
    let symbol: String
    let companyName: String
    let currentPrice: Double
}

@MainActor
class HomeViewModel: ObservableObject {
    @Published var stocks: [Stock] = []
    @Published var isLoading: Bool = false
    @Published var errorMessage: String? = nil

    private let service = StockDataService()

    /// Fetches the latest close prices for a list of symbols.
    func fetchLatestPrices(symbols: [String]) async {
        guard !symbols.isEmpty else {
            self.stocks = []
            return
        }

        isLoading = true
        errorMessage = nil

        do {
            let latestBars = try await service.fetchLatestBars(symbols: symbols)

            // Map the response to Stock models
            var fetchedStocks: [Stock] = []

            for symbol in symbols {
                if let bar = latestBars[symbol] {
                    let price = bar.c
                    let stock = Stock(symbol: symbol, companyName: symbol, currentPrice: price)
                    fetchedStocks.append(stock)
                } else {
                    print("No data for symbol: \(symbol)")
                    // Optionally handle missing data
                }
            }

            // Sort stocks alphabetically
            self.stocks = fetchedStocks.sorted { $0.symbol < $1.symbol }

        } catch {
            self.errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
