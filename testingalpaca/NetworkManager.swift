import Foundation

class StockDataService {
    private let baseURL = "http://127.0.0.1:5000" // Update if necessary

    func fetchStockData(symbol: String,
                        timeframe: String = "1Day",
                        start: String = "2024-01-01T00:00:00Z",
                        end: String = "2024-01-31T00:00:00Z",
                        limit: Int = 10) async throws -> [Bar] {

        guard var urlComponents = URLComponents(string: "\(baseURL)/stock_data") else {
            throw URLError(.badURL)
        }

        urlComponents.queryItems = [
            URLQueryItem(name: "symbol", value: symbol),
            URLQueryItem(name: "timeframe", value: timeframe),
            URLQueryItem(name: "start", value: start),
            URLQueryItem(name: "end", value: end),
            URLQueryItem(name: "limit", value: String(limit))
        ]

        guard let url = urlComponents.url else {
            throw URLError(.badURL)
        }

        // Execute network request
        let (data, response) = try await URLSession.shared.data(from: url)


        // Check HTTP response code
        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode != 200 {
            throw URLError(.badServerResponse)
        }

        // Decode JSON
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(StockDataResponse.self, from: data)

        return decoded.bars?[symbol] ?? []
    }
    
    func fetchLatestBars(symbols: [String], feed: String = "iex", currency: String = "USD") async throws -> [String: Bar] {
        guard !symbols.isEmpty else {
            throw URLError(.badURL)
        }

        guard var urlComponents = URLComponents(string: "\(baseURL)/latest_bars") else {
            throw URLError(.badURL)
        }

        let symbolsParam = symbols.joined(separator: ",")

        urlComponents.queryItems = [
            URLQueryItem(name: "symbols", value: symbolsParam),
            URLQueryItem(name: "feed", value: feed),
            URLQueryItem(name: "currency", value: currency)
        ]

        guard let url = urlComponents.url else {
            throw URLError(.badURL)
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode != 200 {
            throw URLError(.badServerResponse)
        }

        // Decode the response
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(LatestBarsResponse.self, from: data)

        return decoded.bars
    }
    
    func fetchPrediction(symbol: String) async throws -> PredictionResponse{
        guard var urlComponents = URLComponents(string: "\(baseURL)/predict") else {
            throw URLError(.badURL)
        }
        
        urlComponents.queryItems = [
            URLQueryItem(name: "symbol", value: symbol)
        ]
        
        guard let url = urlComponents.url else {
            throw URLError(.badURL)
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PredictionResponse.self, from: data)
        
        return decoded
    }
    
    static func searchStocks(keyword: String) async throws -> [SearchResult] {
            let urlString = "https://ticker-2e1ica8b9.now.sh/keyword/\(keyword)"
            guard let url = URL(string: urlString) else {
                throw URLError(.badURL)
            }
            
            let (data, _) = try await URLSession.shared.data(from: url)
            let results = try JSONDecoder().decode([SearchResult].self, from: data)
            return results
        }

}
