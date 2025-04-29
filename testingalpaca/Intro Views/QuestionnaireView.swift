import SwiftUI


enum NavigationRoute: Hashable {
    case home
}

struct QuestionnaireView: View {
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding: Bool = false
    @StateObject private var viewModel = QuestionnaireViewModel()
    @State private var currentQuestionIndex = 0
    @State private var showCompletionScreen = false
    
    var body: some View {
        ZStack {
            Color(.systemGray5)
                .edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 20) {
                ProgressBar(progress: Float(currentQuestionIndex) / Float(viewModel.questions.count))
                    .frame(height: 8)
                    .padding(.horizontal)
                
                
                if !showCompletionScreen {
                    questionView
                } else {
                    completionView
                }
            }
            .padding()
        }
        .animation(.easeInOut, value: currentQuestionIndex)
        .animation(.easeInOut, value: showCompletionScreen)

        
    }
    
    private var questionView: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Question \(currentQuestionIndex + 1) of \(viewModel.questions.count)")
                .font(.subheadline)
                .foregroundColor(Color.gray)
            
            Text(viewModel.questions[currentQuestionIndex].text)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.black)
            
            responseView
            
            Spacer()
            
            HStack {
                if currentQuestionIndex > 0 {
                    Button(action: {
                        currentQuestionIndex -= 1
                    }) {
                        HStack {
                            Image(systemName: "chevron.left")
                            Text("Previous")
                        }
                        .foregroundColor(Color(.indigo))
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.white)
                        .cornerRadius(10)
                        .shadow(color: Color.black.opacity(0.1), radius: 5)
                    }
                } else {
                    Spacer()
                }
                
                Spacer()
                
                Button(action: {
                    if currentQuestionIndex < viewModel.questions.count - 1 {
                        currentQuestionIndex += 1
                    } else {
                        viewModel.saveResponses()
                        showCompletionScreen = true
                    }
                }) {
                    HStack {
                        Text(currentQuestionIndex < viewModel.questions.count - 1 ? "Next" : "Finish")
                        Image(systemName: "chevron.right")
                    }
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.indigo))
                    .cornerRadius(10)
                    .shadow(color: Color.black.opacity(0.1), radius: 5)
                }
            }
        }
        .padding()
        .background(Color.white)
        .cornerRadius(15)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }
    
    private var responseView: some View {
        let question = viewModel.questions[currentQuestionIndex]
        
        switch question.type {
        case .multiplechoice:
            return AnyView(
                ForEach(0..<question.options.count, id: \.self) { index in
                    Button(action: {
                        viewModel.responses[currentQuestionIndex] = question.options[index]
                    }) {
                        HStack {
                            Text(question.options[index])
                                .foregroundColor(.primary)
                            
                            Spacer()
                            
                            if viewModel.responses[currentQuestionIndex] == question.options[index] {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(Color.indigo)
                            } else {
                                Image(systemName: "circle")
                                    .foregroundColor(Color.gray)
                            }
                        }
                        .padding()
                        .background(Color.white)
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(viewModel.responses[currentQuestionIndex] == question.options[index] ?
                                        Color.indigo : Color.gray.opacity(0.3), lineWidth: 1)
                        )
                    }
                }
            )
        case .slider:
            return AnyView(
                VStack {
                    Slider(value: Binding(
                        get: {
                            let response = viewModel.sliderResponses[currentQuestionIndex] ?? 0
                            return Double(response)
                        },
                        set: { newValue in
                            viewModel.sliderResponses[currentQuestionIndex] = Float(newValue)
                        }
                    ), in: 1.0...50.0, step: 1.0)
                    .accentColor(Color.indigo)
                    
                    HStack {
                        Text("\(Int(viewModel.sliderResponses[currentQuestionIndex] ?? 0))")
                            .foregroundColor(Color.indigo)
                            .fontWeight(.bold)
                        
                        Spacer()
                        
                        Text(question.sliderValueDescription)
                            .foregroundColor(.gray)
                            .font(.caption)
                    }
                    .padding(.horizontal, 8)
                }
            )
        case .textfield:
            return AnyView(
                TextField("Your answer", text: Binding(
                    get: {
                        viewModel.responses[currentQuestionIndex] ?? ""
                    },
                    set: { newValue in
                        viewModel.responses[currentQuestionIndex] = newValue
                    }
                ))
                .padding()
                .background(Color.white)
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                )
            )
        case .datepicker:
            return AnyView(
                DatePicker(
                    "Select a date",
                    selection: Binding(
                        get: {
                            viewModel.dateResponses[currentQuestionIndex] ?? Date()
                        },
                        set: { newValue in
                            viewModel.dateResponses[currentQuestionIndex] = newValue
                        }
                    ),
                    displayedComponents: .date
                )
                .datePickerStyle(GraphicalDatePickerStyle())
                .accentColor(Color.indigo)
            )
        }
    }
    
    private var completionView: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 100, height: 100)
                .foregroundColor(Color.indigo)
            
            Text("Thank You!")
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundColor(Color.gray)
            
            Text("Your financial profile has been created successfully. We'll use this information to provide personalized recommendations tailored just for you.")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundColor(.gray)
                .padding(.horizontal)
            
            Button(action: {
                if let persona = viewModel.generatePersona() {
                    let recommendations = viewModel.generateRecommendations(for: persona)

                    do {
                        let encodedPersona = try JSONEncoder().encode(persona)
                        UserDefaults.standard.set(encodedPersona, forKey: "UserPersona")
                        
                        let encodedRecs = try JSONEncoder().encode(recommendations)
                        UserDefaults.standard.set(encodedRecs, forKey: "InvestmentRecommendations")
                        
                        hasCompletedOnboarding = true
                    } catch {
                        print("Error encoding persona: \(error)")
                    }
                }
            }) {
                Text("Continue to Dashboard")
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.indigo)
                    .cornerRadius(10)
                    .shadow(color: Color.black.opacity(0.1), radius: 5)
            }
            .padding(.top, 20)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(15)
        .shadow(color: Color.black.opacity(0.1), radius: 10)
    }
    
}

struct ProgressBar: View {
    var progress: Float
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Rectangle()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .opacity(0.3)
                    .foregroundColor(Color(.systemGray5))
                
                Rectangle()
                    .frame(width: min(CGFloat(self.progress) * geometry.size.width, geometry.size.width), height: geometry.size.height)
                    .foregroundColor(Color.indigo)
                    .animation(.linear, value: progress)
            }
            .cornerRadius(45)
        }
    }
}
