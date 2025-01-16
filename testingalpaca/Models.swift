import Foundation

struct StockDataResponse: Codable {
    let bars: [String: [Bar]]?
    let next_page_token: String?
}

struct Bar: Codable, Identifiable {
    var id: String {
        return String(t.timeIntervalSince1970)
    }

    let c: Double  // Close
    let h: Double  // High
    let l: Double  // Low
    let n: Int     // Number of trades
    let o: Double  // Open
    let t: Date    // Timestamp as Date
    let v: Int     // Volume
    let vw: Double // Volume Weighted Price

    enum CodingKeys: String, CodingKey {
        case c, h, l, n, o, t, v, vw
    }
}

extension Bar {
    // Custom initializer to parse the date
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        c = try container.decode(Double.self, forKey: .c)
        h = try container.decode(Double.self, forKey: .h)
        l = try container.decode(Double.self, forKey: .l)
        n = try container.decode(Int.self, forKey: .n)
        o = try container.decode(Double.self, forKey: .o)
        v = try container.decode(Int.self, forKey: .v)
        vw = try container.decode(Double.self, forKey: .vw)
        let tString = try container.decode(String.self, forKey: .t)
        
        // Parse the date string to Date without fractional seconds
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime] // Removed .withFractionalSeconds
        if let date = formatter.date(from: tString) {
            t = date
        } else {
            // Throw an error if date parsing fails
            throw DecodingError.dataCorruptedError(forKey: .t, in: container, debugDescription: "Date string does not match format expected by formatter.")
        }
    }
    
    var xLabel: String {
        // Customize as needed for a short label
        let formatter = DateFormatter()
        // e.g., "MMM d h:mm a" => "Jan 3 10:15 AM"
        formatter.dateFormat = "MMM d h:mm a"
        return formatter.string(from: t)
    }

}

struct LatestBarsResponse: Codable {
    let bars: [String: Bar]
}


// Prediction API response
struct PredictionResponse: Codable, Identifiable, Equatable {
    let id = UUID()
    let symbol: String
    let current_price: Double
    let predicted_price: Double
    let predicted_change: Double
    let signal: String
    
    enum CodingKeys: String, CodingKey {
        case symbol
        case current_price
        case predicted_price
        case predicted_change
        case signal
    }
}

// Search
struct SearchResult: Codable, Identifiable {
    let symbol: String
    let name: String
    
    var id: String { symbol }
}
