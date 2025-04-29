import SwiftUI

struct StockDetailView: View {
    // MARK: - Inputs
    let symbol: String
    let companyName: String

    // MARK: - Binding
    @Binding var symbols: [String]

    // MARK: - ViewModel & State
    @StateObject private var viewModel = StockDetailViewModel()
    @State private var selectedRange: ChartRange = .oneDay

    // MARK: - Body
    var body: some View {
        content
            .navigationTitle(symbol)
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(trailing: toggleButton)
            .onAppear(perform: fetchData)
            .onChange(of: selectedRange) { _ in
                viewModel.fetchData(symbol: symbol, chartRange: selectedRange)
            }
    }
}

// MARK: - Subviews
private extension StockDetailView {
    /// The main content of the Stock Detail screen.
    var content: some View {
        VStack(alignment: .leading) {
            header
            chartSection
            rangePicker
            predictionSection
            Spacer()
        }
    }

    /// Displays the stock symbol and company name at the top.
    var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(symbol)
                .font(.title)
                .bold()
            Text(companyName)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.top)
    }

    /// Shows the price and volume chart for the stock.
    var chartSection: some View {
        ChartView(
            bars: viewModel.bars,
            isLoading: viewModel.isLoading,
            errorMessage: viewModel.errorMessage
        )
        .padding(.bottom, 8)
    }

    /// Segmented picker for selecting different chart time ranges (e.g., 1D, 1W, etc.).
    var rangePicker: some View {
        Picker("Time Range", selection: $selectedRange) {
            ForEach(ChartRange.allCases, id: \.self) { range in
                Text(range.rawValue).tag(range)
            }
        }
        .pickerStyle(.segmented)
        .padding()
    }

    /// Displays the loading/error/success states for AI predictions.
    var predictionSection: some View {
        Group {
            if viewModel.isLoadingPrediction {
                ProgressView("Loading AI Prediction...")
                    .frame(maxWidth: .infinity)
                    .padding()
            } else if let errorMessage = viewModel.errorMessage,
                      viewModel.prediction == nil {
                Text("Error: \(errorMessage)")
                    .foregroundColor(.red)
                    .frame(maxWidth: .infinity)
                    .cornerRadius(10)
                    .transition(.opacity)
                    .padding()
            } else if let prediction = viewModel.prediction {
                PredictionView(prediction: prediction)
                    .transition(.move(edge: .bottom))
                    .animation(.spring(), value: prediction)
            }
        }
    }

    /// A button for adding/removing the current symbol to/from the user's list.
    var toggleButton: some View {
        let isSymbolAdded = symbols.contains(symbol)
        return Button {
            if isSymbolAdded {
                symbols.removeAll { $0 == symbol }
            } else {
                symbols.append(symbol)
            }
        } label: {
            Image(systemName: isSymbolAdded ? "minus.circle" : "plus.circle")
                .imageScale(.large)
        }
    }
}

// MARK: - Actions
private extension StockDetailView {
    /// Fetches both chart data and AI prediction for the given symbol.
    func fetchData() {
        viewModel.fetchData(symbol: symbol, chartRange: selectedRange)
        viewModel.fetchPrediction(symbol: symbol)
    }
}

// MARK: - Preview
struct StockDetailView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            StockDetailView(
                symbol: "AAPL",
                companyName: "Apple Inc.",
                symbols: .constant(["AAPL", "TSLA", "GOOG", "AMZN"])
            )
        }
    }
}
