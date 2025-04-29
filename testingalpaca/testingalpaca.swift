//
//  StockTrackerApp.swift
//  testingalpaca
//
//  Created by Varun Kota on 1/10/25.
//


import SwiftUI

@main
struct testingalpaca: App { // Replace `StockTrackerApp` with your app's name
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding: Bool = false
    
    
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                Group {
                    if hasCompletedOnboarding {
                        HomeView()
                    } else {
                        SplashScreen()
                    }
                }
                .animation(.easeInOut, value: hasCompletedOnboarding)
            }
        }
    }
}
