// HomeView.swift
import SwiftUI

struct HomeView: View {
    let symbols = ["AAPL", "TSLA", "GOOG", "AMZN"]
    
    @StateObject private var viewModel = HomeViewModel()
    
    @State private var searchText: String = ""
    @State private var suggestions: [SearchResult] = []

    var body: some View {
        NavigationStack {
            VStack {
                List(viewModel.stocks) { stock in
                    NavigationLink(destination: StockDetailView(symbol: stock.symbol, companyName: stock.companyName)) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(stock.symbol)
                                    .font(.headline)
                                Text(stock.companyName)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Text(String(format: "$%.2f", stock.currentPrice))
                                .font(.headline)
                                .foregroundColor(.primary)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .navigationTitle("Stocks")
                .searchable(text: $searchText, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search stocks")
                .onChange(of: searchText) { query in
                    Task {
                        guard !query.isEmpty else {
                            suggestions = []
                            return
                        }
                        do {
                            suggestions = try await StockDataService.searchStocks(keyword: query)
                        } catch {
                            print("Search error: \(error)")
                            suggestions = []
                        }
                    }
                }
                .searchSuggestions {
                    ForEach(suggestions) { result in
                        NavigationLink(destination: StockDetailView(symbol: result.symbol, companyName: result.name)) {
                            VStack(alignment: .leading) {
                                Text(result.symbol)
                                    .font(.headline)
                                Text(result.name)
                                    .font(.subheadline)
                            }
                        }
                    }
                }
                .onAppear {
                    Task {
                        await viewModel.fetchLatestPrices(symbols: symbols)
                    }
                }
                    .refreshable {
                        Task {
                            await viewModel.fetchLatestPrices(symbols: symbols)
                        }
                    }
                .overlay {
                    if viewModel.isLoading {
                        ProgressView()
                    } else if let errorMessage = viewModel.errorMessage {
                        Text("Error: \(errorMessage)")
                            .foregroundColor(.red)
                    }
                }
            }
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}

