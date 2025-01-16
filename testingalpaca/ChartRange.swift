//
//  ChartRange.swift
//  testingalpaca
//
//  Created by Varun Kota on 1/10/25.
//
import Foundation

enum ChartRange: String, CaseIterable {
    case oneDay  = "1D"
    case fiveDays = "5D"
    case oneMonth = "1M"
    case sixMonths = "6M"
    case ytd = "YTD"
    case oneYear = "1Y"
    case threeYears = "3Y"
    case allTime = "ALL"
    
    /// Display name for segmented picker (if desired)
    var displayName: String {
        switch self {
        case .oneDay:    return "1 Day"
        case .fiveDays:  return "5 Days"
        case .oneMonth:  return "1 Month"
        case .sixMonths: return "6 Months"
        case .ytd:       return "YTD"
        case .oneYear:   return "1 Year"
        case .threeYears: return "3 Years"
        case .allTime:   return "All"
        }
    }
    
    var alpacaTimeframe: String {
        switch self {
        case .oneDay:
            return "5Min"  // Intraday data for the past day
        case .fiveDays:
            return "10Min"  // More granular for a 5-day range
        case .oneMonth:
            return "1Hour"
        case .sixMonths, .ytd:
            return "6Hour"
        case .oneYear:
            return "1Day"
        case .threeYears:
            return "1Week"
        case .allTime:
            return "1Week"
        }
    }
    
    /// Compute the start date (in ISO8601) for each range
    func startDate() -> String {
        let calendar = Calendar.current
        let now = Date()
        
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]
        
        switch self {
        case .oneDay:
            // 24 hours ago
            if let date = calendar.date(byAdding: .day, value: -1, to: now) {
                return isoFormatter.string(from: date)
            }
        case .fiveDays:
            // 5 days ago
            if let date = calendar.date(byAdding: .day, value: -5, to: now) {
                return isoFormatter.string(from: date)
            }
        case .oneMonth:
            // 1 month ago
            if let date = calendar.date(byAdding: .month, value: -1, to: now) {
                return isoFormatter.string(from: date)
            }
        case .sixMonths:
            // 6 months ago
            if let date = calendar.date(byAdding: .month, value: -6, to: now) {
                return isoFormatter.string(from: date)
            }
        case .ytd:
            // From Jan 1 of the current year
            let currentYear = calendar.component(.year, from: now)
            var components = DateComponents()
            components.year = currentYear
            components.month = 1
            components.day = 1
            if let date = calendar.date(from: components) {
                return isoFormatter.string(from: date)
            }
        case .oneYear:
            // 12 months ago
            if let date = calendar.date(byAdding: .year, value: -1, to: now) {
                return isoFormatter.string(from: date)
            }
        case .threeYears:
            // 3 years ago
            if let date = calendar.date(byAdding: .year, value: -3, to: now) {
                return isoFormatter.string(from: date)
            }
        case .allTime:
            // A fixed date sufficiently far in the past
            return "2000-01-01T00:00:00Z"
        }
        
        // Fallback to now if something fails
        return isoFormatter.string(from: now)
    }
}
