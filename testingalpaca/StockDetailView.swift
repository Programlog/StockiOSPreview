import SwiftUI

struct StockDetailView: View {
    let symbol: String
    let companyName: String
    
    @StateObject private var viewModel = StockDetailViewModel()
    @State private var selectedRange: ChartRange = .oneDay
    
    var body: some View {
        VStack(alignment: .leading) {
            // Symbol & Company Name
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

            // Combined Chart (Price + Volume)
            ChartView(
                bars: viewModel.bars,
                isLoading: viewModel.isLoading,
                errorMessage: viewModel.errorMessage
            )
            .padding(.bottom, 8)

            // Time Range Picker
            Picker("Time Range", selection: $selectedRange) {
                ForEach(ChartRange.allCases, id: \.self) { range in
                    Text(range.rawValue).tag(range)
                }
            }
            .pickerStyle(.segmented)
            .padding()
            
            if viewModel.isLoadingPrediction {
                ProgressView("Loading AI Prediction...")
                    .frame(maxWidth: .infinity)
                    .padding()
            } else if let errorMessage = viewModel.errorMessage, viewModel.prediction == nil {
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

            Spacer()
        }
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            viewModel.fetchData(symbol: symbol, chartRange: selectedRange)
            viewModel.fetchPrediction(symbol: symbol)
        }
        .onChange(of: selectedRange) { newRange in
            viewModel.fetchData(symbol: symbol, chartRange: newRange)
        }
    }
}

struct StockDetailView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            StockDetailView(
                symbol: "AAPL",
                companyName: "Apple Inc."
            )
        }
    }
}
