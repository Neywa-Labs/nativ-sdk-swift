// MiniCPM-OCR preprocessing follows OpenBMB's Apache-2.0 reference processor.
// See Documentation/Models/MiniCPMOCR/README.md for attribution.
import CoreImage
import Foundation
import MLX
import MLXNN

/// Shift Qwen norm scales in FP32 while keeping the activation stream in its input dtype.
final class MiniCPMOCRRMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x).asType(x.dtype)
    }
}

struct MiniCPMOCRImageOptions: Decodable, Sendable {
    var disableSmallImageUpsize = false
    var adaptiveSmallImageUpsize = false
    var fontPixels: Double = 12
    var componentsPerKilopixel: Double = 2
    var tinyPixels = 6272
    var tinyResolution = 84
    var adaptiveTileUpsize = false

    enum CodingKeys: String, CodingKey {
        case disableSmallImageUpsize = "disable_small_image_upsize"
        case adaptiveSmallImageUpsize = "adaptive_small_image_upsize"
        case fontPixels = "adaptive_upsize_font_px"
        case componentsPerKilopixel = "adaptive_upsize_dense_per_kpx"
        case tinyPixels = "adaptive_upsize_tiny_pixels"
        case tinyResolution = "adaptive_upsize_tiny_target_resolution"
        case adaptiveTileUpsize = "adaptive_tile_upsize"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        disableSmallImageUpsize =
            try c.decodeIfPresent(Bool.self, forKey: .disableSmallImageUpsize) ?? false
        adaptiveSmallImageUpsize =
            try c.decodeIfPresent(Bool.self, forKey: .adaptiveSmallImageUpsize) ?? false
        fontPixels = try c.decodeIfPresent(Double.self, forKey: .fontPixels) ?? 12
        componentsPerKilopixel =
            try c.decodeIfPresent(Double.self, forKey: .componentsPerKilopixel) ?? 2
        tinyPixels = try c.decodeIfPresent(Int.self, forKey: .tinyPixels) ?? 6272
        tinyResolution = try c.decodeIfPresent(Int.self, forKey: .tinyResolution) ?? 84
        adaptiveTileUpsize = try c.decodeIfPresent(Bool.self, forKey: .adaptiveTileUpsize) ?? false
    }

    mutating func apply(_ context: [String: any Sendable]) throws {
        disableSmallImageUpsize = try Self.value(
            "disable_small_image_upsize", in: context, default: disableSmallImageUpsize)
        adaptiveSmallImageUpsize = try Self.value(
            "adaptive_small_image_upsize", in: context, default: adaptiveSmallImageUpsize)
        fontPixels = try Self.number("adaptive_upsize_font_px", in: context, default: fontPixels)
        componentsPerKilopixel = try Self.number(
            "adaptive_upsize_dense_per_kpx", in: context, default: componentsPerKilopixel)
        tinyPixels = try Self.value("adaptive_upsize_tiny_pixels", in: context, default: tinyPixels)
        tinyResolution = try Self.value(
            "adaptive_upsize_tiny_target_resolution", in: context, default: tinyResolution)
        adaptiveTileUpsize = try Self.value(
            "adaptive_tile_upsize", in: context, default: adaptiveTileUpsize)
        guard fontPixels.isFinite, fontPixels > 0, componentsPerKilopixel.isFinite,
            componentsPerKilopixel >= 0, tinyPixels > 0, tinyResolution > 0
        else {
            throw VLMError.processing(
                "MiniCPM-OCR adaptive image settings must be finite and positive")
        }
    }

    static func value<T>(_ name: String, in context: [String: any Sendable], default fallback: T)
        throws -> T
    {
        guard let raw = context[name] else { return fallback }
        guard let value = raw as? T else {
            throw VLMError.processing("MiniCPM-OCR invalid processor option: \(name)")
        }
        return value
    }

    private static func number(
        _ name: String, in context: [String: any Sendable], default fallback: Double
    ) throws -> Double {
        guard let raw = context[name] else { return fallback }
        if let value = raw as? Double { return value }
        if let value = raw as? Int { return Double(value) }
        if let value = raw as? Float { return Double(value) }
        throw VLMError.processing("MiniCPM-OCR invalid processor option: \(name)")
    }

    enum Tier: Equatable { case full, tiny, native }

    func tier(_ image: CIImage) -> Tier {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let rgb = MediaProcessing.renderRGBArray(image).asArray(Float.self)
        var gray = [UInt8]()
        gray.reserveCapacity(width * height)
        for i in stride(from: 0, to: rgb.count, by: 3) {
            let r = Int((min(max(rgb[i], 0), 1) * 255).rounded())
            let g = Int((min(max(rgb[i + 1], 0), 1) * 255).rounded())
            let b = Int((min(max(rgb[i + 2], 0), 1) * 255).rounded())
            // Pillow's RGB -> L coefficients and integer rounding.
            gray.append(UInt8((19595 * r + 38470 * g + 7471 * b + 32768) >> 16))
        }
        return tier(gray: gray, width: width, height: height)
    }

    /// Otsu thresholding and 8-connected components, matching the reference's OpenCV policy.
    func tier(gray: [UInt8], width: Int, height: Int) -> Tier {
        guard width > 0, height > 0, gray.count == width * height else { return .native }
        var histogram = [Int](repeating: 0, count: 256)
        for value in gray { histogram[Int(value)] += 1 }
        let total = gray.count
        let sum = histogram.enumerated().reduce(0.0) { $0 + Double($1.offset * $1.element) }
        var background = 0
        var backgroundSum = 0.0
        var bestVariance = 0.0
        var threshold = 0
        for value in 0 ..< 256 {
            background += histogram[value]
            backgroundSum += Double(value * histogram[value])
            guard background > 0, background < total else { continue }
            let foreground = total - background
            let difference =
                backgroundSum / Double(background) - (sum - backgroundSum) / Double(foreground)
            let variance = Double(background) * Double(foreground) * difference * difference
            if variance > bestVariance {
                bestVariance = variance
                threshold = value
            }
        }
        let darkCount = gray.reduce(0) { $0 + (Int($1) <= threshold ? 1 : 0) }
        let inverted = darkCount > total / 2
        var visited = [Bool](repeating: false, count: total)
        var queue = [Int]()
        var heights = [Int]()
        for start in 0 ..< total {
            guard !visited[start], (Int(gray[start]) <= threshold) != inverted else { continue }
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            visited[start] = true
            var cursor = 0
            var minX = start % width
            var maxX = minX
            var minY = start / width
            var maxY = minY
            while cursor < queue.count {
                let index = queue[cursor]
                cursor += 1
                let x = index % width
                let y = index / width
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
                for ny in max(0, y - 1) ... min(height - 1, y + 1) {
                    for nx in max(0, x - 1) ... min(width - 1, x + 1) {
                        let next = ny * width + nx
                        if !visited[next], (Int(gray[next]) <= threshold) != inverted {
                            visited[next] = true
                            queue.append(next)
                        }
                    }
                }
            }
            let componentWidth = maxX - minX + 1
            let componentHeight = maxY - minY + 1
            if queue.count >= 3, componentHeight >= 2,
                Double(componentWidth) < 0.9 * Double(width),
                Double(componentHeight) < 0.9 * Double(height)
            {
                heights.append(componentHeight)
            }
        }
        heights.sort()
        let median: Double
        if heights.isEmpty {
            median = 0
        } else if heights.count.isMultiple(of: 2) {
            median = Double(heights[heights.count / 2 - 1] + heights[heights.count / 2]) / 2
        } else {
            median = Double(heights[heights.count / 2])
        }
        let density = Double(heights.count) * 1000 / Double(total)
        if median > 0, median <= fontPixels, density >= componentsPerKilopixel { return .full }
        return total < tinyPixels ? .tiny : .native
    }
}

extension MiniCPMVProcessorConfiguration {
    func applyingOCROptions(_ context: [String: any Sendable]) throws -> Self {
        var result = self
        result.downsampleMode = try MiniCPMOCRImageOptions.value(
            "downsample_mode", in: context, default: downsampleMode)
        result.maxSliceNums = try MiniCPMOCRImageOptions.value(
            "max_slice_nums", in: context, default: maxSliceNums)
        result.sliceMode = try MiniCPMOCRImageOptions.value(
            "slice_mode", in: context, default: sliceMode)
        result.useImageID = try MiniCPMOCRImageOptions.value(
            "use_image_id", in: context, default: useImageID)
        try result.ocrImageOptions.apply(context)
        guard ["4x", "16x"].contains(result.downsampleMode), result.patchSize > 0,
            result.scaleResolution > 0, result.maxSliceNums >= 0, result.imageMean.count == 3,
            result.imageStd.count == 3, result.imageStd.allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw VLMError.processing(
                "MiniCPM-OCR requires valid geometry and downsample_mode '4x' or '16x'")
        }
        return result
    }
}

/// The prepared crop spans carry the per-request compression mode without mutable model state.
func miniCPMOCRDownsampleMode(frames: [THW], bounds: MLXArray?) throws -> String {
    guard let bounds, bounds.shape == [frames.count, 2], !frames.isEmpty else {
        throw VLMError.processing("MiniCPM-OCR requires one placeholder span per image crop")
    }
    let values = bounds.asArray(Int.self)
    var divisor: Int?
    for (index, frame) in frames.enumerated() {
        let count = values[index * 2 + 1] - values[index * 2]
        guard count > 0, frame.h > 0, frame.w > 0, (frame.h * frame.w).isMultiple(of: count) else {
            throw VLMError.processing("MiniCPM-OCR invalid crop placeholder count")
        }
        let current = frame.h * frame.w / count
        let factor = current == 4 ? 2 : 4
        guard [4, 16].contains(current), frame.h.isMultiple(of: factor),
            frame.w.isMultiple(of: factor),
            divisor == nil || divisor == current
        else {
            throw VLMError.processing(
                "MiniCPM-OCR crop compression does not match placeholder counts")
        }
        divisor = current
    }
    return divisor == 4 ? "4x" : "16x"
}

/// Float32 boundary comparisons reproduce SigLIP's trained position buckets.
func miniCPMVPositionBuckets(count: Int, gridSide: Int) -> [Int] {
    let boundaries = (1 ..< gridSide).map { Float(Double($0) / Double(gridSide)) }
    return (0 ..< count).map { index in
        let fraction = Float(index) / Float(count)
        var low = 0
        var high = boundaries.count
        while low < high {
            let middle = (low + high) / 2
            if boundaries[middle] <= fraction { low = middle + 1 } else { high = middle }
        }
        return low
    }
}
