struct UserPersona: Codable {
    enum RiskTolerance: String, Codable {
        case veryConservative, conservative, moderate, aggressive, veryAggressive
    }
    
    enum FinancialCapacity: String, Codable {
        case low, medium, high, veryHigh
    }
    
    enum DebtLevel: String, Codable {
        case none, low, moderate, high
    }
    
    // Properties
    let riskTolerance: RiskTolerance
    let financialCapacity: FinancialCapacity
    let retirementTimeHorizon: Int // Years until retirement
    let investmentStrategy: String
    let retirementGoals: String
    let debtLevel: DebtLevel
    let expectedLifestyle: String
    let healthStatus: String
    let plannedRetirementAge: Int
}

