//
//  SearchBar.swift
//  testingalpaca
//
//  Created by Varun Kota on 1/15/25.
import SwiftUI


struct SearchBar: View {
    @Binding var text: String
    @Binding var suggestions: [SearchResult]
    @Binding var isSearching: Bool
    
    var body: some View {
        HStack {
            TextField("Search stocks...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .onChange(of: text) { newValue in
                    guard !newValue.isEmpty else {
                        suggestions = []
                        isSearching = false
                        return
                    }
                    
                    Task {
                        do {
                            suggestions = try await StockDataService.searchStocks(keyword: newValue)
                            isSearching = true
                        } catch {
                            print("Search error: \(error)")
                            suggestions = []
                        }
                    }
                }
            
            if !text.isEmpty {
                Button(action: {
                    text = ""
                    suggestions = []
                    isSearching = false
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.gray)
                }
            }
        }
    }
}
