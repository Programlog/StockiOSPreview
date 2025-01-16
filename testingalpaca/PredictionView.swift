//
//  PredictionView.swift
//  testingalpaca
//
//  Created by Varun Kota on 1/15/25.
//


// PredictionView.swift
import SwiftUI

struct PredictionView: View {
    let prediction: PredictionResponse
    
    @State private var animateChange: Bool = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("AI Prediction")
                .font(.headline)
                .foregroundColor(.primary)
            
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Current Price:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "$%.2f", prediction.current_price))
                            .font(.subheadline)
                            .foregroundColor(.primary)
                    }
                    
                    HStack {
                        Text("Predicted Price:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "$%.2f", prediction.predicted_price))
                            .font(.subheadline)
                            .foregroundColor(.primary)
                    }
                    
                    HStack {
                        Text("Predicted Change:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "%.2f%%", prediction.predicted_change))
                            .font(.subheadline)
                            .foregroundColor(prediction.predicted_change >= 0 ? .green : .red)
                    }
                }
                .padding()
                .background(Color(UIColor.systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 5)
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.5), value: prediction)
                
                Spacer()
            }
            
            HStack {
                Text("Signal:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text(prediction.signal.uppercased())
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(colorForSignal(prediction.signal))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    .shadow(color: Color.black.opacity(0.15), radius: 3, x: 0, y: 3)
                    .animation(.easeInOut(duration: 0.3), value: prediction.signal)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(UIColor.secondarySystemBackground))
                .shadow(color: Color.black.opacity(0.1), radius: 10, x: 0, y: 5)
        )
        .padding(.horizontal)
        .scaleEffect(animateChange ? 1.05 : 1.0)
        .onAppear {
            withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
                animateChange.toggle()
            }
        }
    }
    
    private func colorForSignal(_ signal: String) -> Color {
        switch signal.uppercased() {
        case "BUY":
            return .green
        case "SELL":
            return .red
        default:
            return .gray
        }
    }
}

struct PredictionView_Previews: PreviewProvider {
    static var previews: some View {
        PredictionView(prediction: PredictionResponse(
            symbol: "AAPL",
            current_price: 150.00,
            predicted_price: 155.00,
            predicted_change: 3.33,
            signal: "BUY"
        ))
            .preferredColorScheme(.light)
    }
}