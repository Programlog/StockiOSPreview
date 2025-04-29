// HomeView.swift
import SwiftUI


struct HomeView: View {
    // MARK: - State & ViewModel
    @State private var symbols = ["AAPL", "TSLA", "GOOG", "AMZN"]
    @StateObject private var viewModel = HomeViewModel()
    
    @State private var searchText: String = ""
    @State private var suggestions: [SearchResult] = []
    
    @StateObject private var questionViewModel = QuestionnaireViewModel()
    @State private var persona: UserPersona?
    @State private var recommendations: InvestmentRecommendation?
    @State private var showingRecommendations = false
    
    @State private var isLoadingData = false



    
    // MARK: - Body
    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Stocks")
                .searchable(
                    text: $searchText,
                    placement: .navigationBarDrawer(displayMode: .automatic),
                    prompt: "Search stocks"
                )
                .searchSuggestions {
                    suggestionsView
                }
                .task {
                    // Load the latest prices when first displayed
                    await loadLatestPrices()
                }
                .refreshable {
                    // Pull-to-refresh
                    await loadLatestPrices()
                }
                .overlay(overlayView)
                .onChange(of: searchText, perform: handleSearchTextChange)
        }
    }
}

// MARK: - Subviews
private extension HomeView {
    /// The main content of the HomeView
    var content: some View {
        VStack {
            stocksList
            
            Spacer()
            
            
            
            Button {
                showingRecommendations = true
            } label: {
                Image(systemName: "list.bullet.rectangle")
                    .font(.title2)
                    .padding()
                    .background(Color.indigo)
                    .foregroundColor(.white)
                    .clipShape(Circle())
                    .shadow(radius: 5)
            }
            .padding(.bottom)

        }
        .sheet(isPresented: $showingRecommendations) {
                    Group {
                        if let persona = persona, let recommendations = recommendations {
                            RecommendationsView(
                                persona: persona,
                                recommendations: recommendations,
                                viewModel: questionViewModel
                            )
                        } else {
                            ProgressView("Loading recommendations...")
                                .onAppear(perform: loadUserData)
                        }
                    }
                }
                .onAppear {
                    if !isLoadingData {
                        loadUserData()
                    }
                }

    }
    
    private func loadUserData() {
        isLoadingData = true
        defer { isLoadingData = false }
        
        do {
            if let personaData = UserDefaults.standard.data(forKey: "UserPersona") {
                persona = try JSONDecoder().decode(UserPersona.self, from: personaData)
            }
            
            if let recsData = UserDefaults.standard.data(forKey: "InvestmentRecommendations") {
                recommendations = try JSONDecoder().decode(InvestmentRecommendation.self, from: recsData)
            }
        } catch {
            print("Error loading data: \(error)")
        }
    }


    
    /// Displays a list of stocks from `viewModel.stocks`
    var stocksList: some View {
        List(viewModel.stocks) { stock in
            NavigationLink(destination: StockDetailView(
                symbol: stock.symbol,
                companyName: stock.companyName,
                symbols: $symbols
            )) {
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
                }
                .padding(.vertical, 4)
            }
        }
    }
    
    /// Displays search suggestions for the current `searchText`
    var suggestionsView: some View {
        ForEach(suggestions) { result in
            HStack(spacing: 0) {
                // Tap on the left side navigates to detail view
                NavigationLink(destination: StockDetailView(
                    symbol: result.symbol,
                    companyName: result.name,
                    symbols: $symbols
                )) {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(result.symbol)
                                .font(.headline)
                            Text(result.name)
                                .font(.subheadline)
                        }
                        Spacer()
                    }
                    .contentShape(Rectangle())  // Makes entire area tappable
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
                
                // Plus/Minus button on the right side
                let isInList = symbols.contains(result.symbol)
                Button {
                    if isInList {
                        symbols.removeAll { $0 == result.symbol }
                    } else {
                        symbols.append(result.symbol)
                    }
                } label: {
                    Label(
                        isInList ? "Remove \(result.symbol)" : "Add \(result.symbol)",
                        systemImage: isInList ? "minus.circle" : "plus.circle"
                    )
                    .labelStyle(.iconOnly)
                    .foregroundColor(isInList ? .red : .blue)
                }
                .buttonStyle(.plain)
                .padding(.trailing)
            }
        }
    }
    
    /// Displays an overlay with either a loading indicator or an error message
    var overlayView: some View {
        Group {
            if viewModel.isLoading {
                ProgressView()
            } else if let errorMessage = viewModel.errorMessage {
                Text("Error: \(errorMessage)")
                    .foregroundColor(.red)
            }
        }
    }
}

// MARK: - Actions
private extension HomeView {
    /// Loads the latest prices from the HomeViewModel
    func loadLatestPrices() async {
        await viewModel.fetchLatestPrices(symbols: symbols)
    }
    
    /// Handles changes to the search text by fetching suggestions
    func handleSearchTextChange(_ query: String) {
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
}
struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}

