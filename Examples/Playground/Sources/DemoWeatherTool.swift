import Foundation
import NativSDK

enum DemoWeatherTool {
    struct Input: Codable, Sendable {
        let city: String
    }

    struct Output: Codable, Sendable {
        let city: String
        let temperatureCelsius: Int
        let condition: String
        let isMock: Bool
    }

    static let tool = Tool<Input, Output>(
        name: "get_weather",
        description:
            "Get simulated weather for a city. This demo returns fixed mock data, not real weather.",
        parameters: [
            .required("city", type: .string, description: "The city to get weather for.")
        ]
    ) { input in
        let city = input.city.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !city.isEmpty else {
            throw DemoError("The weather tool requires a nonempty city.")
        }
        return Output(city: city, temperatureCelsius: 21, condition: "Sunny", isMock: true)
    }
}

struct DemoError: LocalizedError {
    let errorDescription: String?

    init(_ description: String) {
        errorDescription = description
    }
}
