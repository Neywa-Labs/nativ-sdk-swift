// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "nativ-sdk-swift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "NativSDK",
            targets: ["NativSDK"],
        ),
        .executable(
            name: "nativ",
            targets: ["nativ"],
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.33.0"),
        ),
        .package(
            url: "https://github.com/Neywa-Labs/mlx-audio-swift",
            .upToNextMinor(from: "0.2.1"),
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            .upToNextMinor(from: "1.0.1"),
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.4.0"),
        ),
    ],
    targets: [
        .target(
            name: "NativSDK",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/NativSDK",
        ),
        .executableTarget(
            name: "nativ",
            dependencies: [
                "NativSDK",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Tools/nativ",
            exclude: [
                "README.md"
            ],
        ),
        .testTarget(
            name: "NativSDKTests",
            dependencies: [
                "NativSDK",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/NativSDKTests",
            resources: [.copy("Fixtures")],
        ),
    ],
)
