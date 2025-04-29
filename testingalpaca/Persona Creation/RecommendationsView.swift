//
//  RecommendationsView.swift
//  testingalpaca
//
//  Created by Varun Kota on 3/5/25.
//

import Foundation
import SwiftUI

struct RecommendationsView: View {
    let persona: UserPersona
    let recommendations: InvestmentRecommendation
    @ObservedObject var viewModel: QuestionnaireViewModel
    @State private var watchlist: [String] = []
    @State private var showWatchlist = false

    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 25) {
                // Risk Profile Section
                VStack(alignment: .leading) {
                    Text("Risk Profile")
                        .font(.headline)
                    HStack {
                        Text("Tolerance:")
                        Text(persona.riskTolerance.rawValue.capitalized)
                            .foregroundColor(.white)
                            .padding(5)
                            .background(Color.blue)
                            .cornerRadius(5)
                        
                        Text("Capacity:")
                        Text(persona.financialCapacity.rawValue.capitalized)
                            .foregroundColor(.white)
                            .padding(5)
                            .background(Color.green)
                            .cornerRadius(5)
                    }
                }
                
                // Allocation Section
                VStack(alignment: .leading) {
                    Text("Suggested Allocation")
                        .font(.headline)
                    Text(recommendations.portfolioAllocation)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                }
                
                // Products Section
                VStack(alignment: .leading) {
                    Text("Recommended Products")
                        .font(.headline)
                    ForEach(recommendations.suggestedProducts, id: \.self) { product in
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text(product)
                        }
                    }
                }
                
                // Special Notes
                if let debtNote = recommendations.debtAdvice {
                    VStack(alignment: .leading) {
                        Text("Debt Advisory")
                            .font(.headline)
                            .foregroundColor(.red)
                        Text(debtNote)
                    }
                }
                
                if showWatchlist && !watchlist.isEmpty {
                    VStack(alignment: .leading) {
                        Text("Curated Watchlist")
                            .font(.headline)
                        
                        ForEach(watchlist, id: \.self) { symbol in
                            HStack {
                                Text(symbol)
                                    .font(.system(.body, design: .monospaced))
                                Spacer()
                                Image(systemName: "chart.line.uptrend.xyaxis")
                                    .foregroundColor(.blue)
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                        }
                    }
                    .transition(.opacity)
                }
                
                // Generate Watchlist Button
                Button {
                    withAnimation {
                        watchlist = viewModel.generateWatchlist(for: persona)
                        showWatchlist = true
                    }
                } label: {
                    Text(showWatchlist ? "Refresh Watchlist" : "Generate Watchlist")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.indigo)
                        .cornerRadius(10)
                }
            }
            .padding()
        }
        .navigationTitle("Your Custom Plan")
    }
}
