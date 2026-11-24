import Foundation
import MLX

/// Fixed reference outputs for cache and generation regression tests.
// Captured from mlx-swift-lm c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8
// using MLX Swift 0.32.2 on Apple silicon/Metal. Keep these expected values
// independent of the implementation under test when updating them.
enum RuntimeReferenceFixtures {
    static func values<T: Decodable>(_ key: String, as type: T.Type = T.self) throws -> T {
        let url = Bundle.module.url(
            forResource: "runtime-reference", withExtension: "json", subdirectory: "Fixtures")!
        let entries =
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        return try JSONDecoder().decode(
            T.self, from: JSONSerialization.data(withJSONObject: entries[key]!))
    }

    static func tensor(_ key: String, index: Int = 0, shape: [Int]) throws -> MLXArray {
        let values: [[Float]] = try values(key)
        return MLXArray(values[index], shape)
    }
}
