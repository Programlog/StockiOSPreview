//
//  QuestionnaireViewModel.swift
//  testingalpaca
//
//  Created by Varun Kota on 3/3/25.
//

import Foundation
import SwiftUICore



class QuestionnaireViewModel: ObservableObject {
    struct Question {
        let text: String
        let type: QuestionType
        let options: [String]
        let sliderValueDescription: String
        
        init(text: String, type: QuestionType, options: [String] = [], sliderValueDescription: String = "") {
            self.text = text
            self.type = type
            self.options = options
            self.sliderValueDescription = sliderValueDescription
        }
    }
    
    enum QuestionType {
        case multiplechoice
        case slider
        case textfield
        case datepicker
    }
    
    let questions: [Question] = [
        Question(text: "What is your annual income bracket?", type: .multiplechoice, options: [
            "Under $30,000",
            "$30,000 - $60,000",
            "$60,001 - $100,000",
            "$100,001 - $150,000",
            "Over $150,000"
        ]),
        Question(text: "How would you describe your risk tolerance for investments?", type: .multiplechoice, options: [
            "Very Conservative",
            "Conservative",
            "Moderate",
            "Aggressive",
            "Very Aggressive"
        ]),
        Question(text: "How many years until you plan to retire?", type: .slider, sliderValueDescription: "years"),
        Question(text: "What are your current savings? (in dollars)", type: .textfield),
        Question(text: "What is your preferred investment strategy?", type: .multiplechoice, options: [
            "Passive Index Funds",
            "Active Management",
            "Mix of Both",
            "Real Estate Focus",
            "Alternative Investments",
            "Not Sure Yet"
        ]),
        Question(text: "What are your primary retirement goals?", type: .multiplechoice, options: [
            "Travel Extensively",
            "Maintain Current Lifestyle",
            "Downsize and Simplify",
            "Start a New Business/Venture",
            "Support Family Members",
            "Charitable Giving"
        ]),
        Question(text: "What is your current debt obligation level?", type: .multiplechoice, options: [
            "No Debt",
            "Low (under 20% of income)",
            "Moderate (20-40% of income)",
            "High (above 40% of income)"
        ]),
        Question(text: "What kind of lifestyle do you expect in retirement?", type: .multiplechoice, options: [
            "Minimalist",
            "Comfortable",
            "Affluent",
            "Luxurious"
        ]),
        Question(text: "How would you rate your overall health status?", type: .multiplechoice, options: [
            "Excellent",
            "Good",
            "Average",
            "Fair",
            "Poor"
        ]),
        Question(text: "At what age do you plan to retire?", type: .datepicker)
    ]
    
    @Published var responses: [Int: String] = [:]
    @Published var sliderResponses: [Int: Float] = [:]
    @Published var dateResponses: [Int: Date] = [:]
    
    init() {
        // Initialize with default values
        for (index, question) in questions.enumerated() {
            switch question.type {
            case .multiplechoice:
                responses[index] = ""
            case .slider:
                sliderResponses[index] = 0
            case .textfield:
                responses[index] = ""
            case .datepicker:
                dateResponses[index] = Date()
            }
        }
    }
    
    func saveResponses() {
        // Create a dictionary to store all response types
        var allResponses: [String: Any] = [:]
        
        for (index, question) in questions.enumerated() {
            switch question.type {
            case .multiplechoice, .textfield:
                if let response = responses[index] {
                    allResponses[question.text] = response
                }
            case .slider:
                if let response = sliderResponses[index] {
                    allResponses[question.text] = response
                }
            case .datepicker:
                if let response = dateResponses[index] {
                    let formatter = DateFormatter()
                    formatter.dateStyle = .medium
                    allResponses[question.text] = formatter.string(from: response)
                }
            }
        }
        
        // Save responses securely
        saveToUserDefaults(allResponses)
        
        // In a real app, you might want to:
        // 1. Encrypt the data
        // 2. Use Keychain for sensitive information
        // 3. Send to a secure backend server
    }
    
    private func saveToUserDefaults(_ responses: [String: Any]) {
        // Convert to Data for storage
        guard let jsonData = try? JSONSerialization.data(withJSONObject: responses) else {
            print("Error: Unable to convert responses to JSON")
            return
        }
        
        // Store in UserDefaults (for demonstration purposes)
        // In a production app, use more secure storage methods for financial data
        UserDefaults.standard.set(jsonData, forKey: "FinancialQuestionnaireResponses")
        UserDefaults.standard.synchronize()
        
        print("Responses saved successfully")
    }
    
    func loadSavedResponses() -> [String: Any]? {
        if let jsonData = UserDefaults.standard.data(forKey: "FinancialQuestionnaireResponses"),
           let responses = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
            return responses
        }
        return nil
    }

    func generatePersona() -> UserPersona? {
        guard let riskAnswer = responses[1], // Index 1 = risk tolerance
              let retirementYears = sliderResponses[2], // Index 2 = years until retirement
              let savingsText = responses[3], // Index 3 = savings
              let investmentStrategy = responses[4], // Index 4 = strategy
              let retirementGoals = responses[5], // Index 5 = goals
              let debtLevel = responses[6], // Index 6 = debt
              let lifestyle = responses[7], // Index 7 = lifestyle
              let health = responses[8], // Index 8 = health
              let retirementDate = dateResponses[9] // Index 9 = retirement age
        else { return nil }
        
        // Calculate financial capacity based on income (index 0) and savings
        let incomeBracket = responses[0] ?? ""
        let savings = Float(savingsText) ?? 0
        let financialCapacity = calculateFinancialCapacity(income: incomeBracket, savings: savings)
        
        // Calculate retirement age
        let plannedAge = Calendar.current.component(.year, from: retirementDate)
        
        return UserPersona(
            riskTolerance: .init(rawValue: riskAnswer.lowercased().replacingOccurrences(of: " ", with: "")) ?? .moderate,
            financialCapacity: financialCapacity,
            retirementTimeHorizon: Int(retirementYears),
            investmentStrategy: investmentStrategy,
            retirementGoals: retirementGoals,
            debtLevel: .init(rawValue: debtLevel.lowercased()) ?? .none,
            expectedLifestyle: lifestyle,
            healthStatus: health,
            plannedRetirementAge: plannedAge
        )
    }
    
    private func calculateFinancialCapacity(income: String, savings: Float) -> UserPersona.FinancialCapacity {
        switch income {
        case "Under $30,000":
            return savings > 5000 ? .medium : .low
        case "$30,000 - $60,000":
            return savings > 15000 ? .medium : .low
        case "$60,001 - $100,000":
            return savings > 30000 ? .high : .medium
        case "$100,001 - $150,000":
            return .high
        case "Over $150,000":
            return .veryHigh
        default:
            return .medium
        }
    }
}


struct InvestmentRecommendation: Codable {
    let portfolioAllocation: String
    let suggestedProducts: [String]
    let riskNote: String
    let debtAdvice: String?
}

extension QuestionnaireViewModel {
    func generateRecommendations(for persona: UserPersona) -> InvestmentRecommendation {
        var allocation = ""
        var products = [String]()
        var riskNote = ""
        var debtAdvice: String? = nil
        
        // Risk-based recommendations
        switch persona.riskTolerance {
        case .veryConservative:
            allocation = "80% Bonds, 15% CDs, 5% Cash"
            products = ["Treasury Bonds", "Money Market Accounts", "CD Ladders"]
            riskNote = "Capital preservation focus"
            
        case .conservative:
            allocation = "60% Bonds, 30% Dividend Stocks, 10% REITs"
            products = ["Bond ETFs", "Blue Chip Stocks", "Real Estate ETFs"]
            
        case .moderate:
            allocation = "50% Stocks, 40% Bonds, 10% Alternatives"
            products = ["Index Funds", "Corporate Bonds", "Balanced Mutual Funds"]
            
        case .aggressive:
            allocation = "70% Stocks, 20% Crypto, 10% Venture Capital"
            products = ["Growth Stocks", "Sector ETFs", "Crypto Index Funds"]
            
        case .veryAggressive:
            allocation = "90% Growth Stocks, 10% Derivatives"
            products = ["Tech Stocks", "LEAP Options", "Leveraged ETFs"]
        }
        
        // Time horizon adjustments
        if persona.retirementTimeHorizon < 10 {
            allocation = "\(allocation) + 5% Annuity Allocation"
            products.append("Fixed Index Annuities")
        }
        
        // Debt considerations
        if persona.debtLevel == .high {
            debtAdvice = "Recommend paying off high-interest debt (above 6%) before aggressive investments"
        }
        
        // Financial capacity additions
        switch persona.financialCapacity {
        case .veryHigh:
            products.append("Private Equity Opportunities")
        case .high:
            products.append("Real Estate Crowdfunding")
        default:
            break
        }
        
        return InvestmentRecommendation(
            portfolioAllocation: allocation,
            suggestedProducts: products,
            riskNote: riskNote,
            debtAdvice: debtAdvice
        )
    }
    
    
    func generateWatchlist(for persona: UserPersona) -> [String] {
        var watchlist: [String] = []
        
        // Risk-based assets
        switch persona.riskTolerance {
        case .veryConservative:
            watchlist += ["BND", "TIP", "GOVT", "MUB"]
        case .conservative:
            watchlist += ["VIG", "VNQ", "SPHD", "PFF"]
        case .moderate:
            watchlist += ["VTI", "BND", "VXUS", "VNQ"]
        case .aggressive:
            watchlist += ["QQQ", "ARKK", "SOXX", "VGT"]
        case .veryAggressive:
            watchlist += ["TSLA", "NVDA", "MSTR", "COIN"]
        }
        
        // Time horizon additions
        if persona.retirementTimeHorizon > 20 {
            watchlist += ["BTC-USD", "ETH-USD", "GBTC"]
        }
        
        // Financial capacity additions
        switch persona.financialCapacity {
        case .high, .veryHigh:
            watchlist += ["PRIVATE", "REIT", "VC"]
        default:
            break
        }
        
        // Debt considerations
        if persona.debtLevel == .high {
            watchlist = watchlist.filter { !["MSTR", "COIN"].contains($0) }
        }
        
        return Array(Set(watchlist)).sorted() // Remove duplicates
    }

}
