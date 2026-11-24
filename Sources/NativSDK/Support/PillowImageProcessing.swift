import Foundation

/// Byte-accurate resampling shared by native document processors.
enum PillowImageProcessing {
    /// Pillow's separable bilinear resize: antialiased triangle taps and 22-bit coefficients.
    /// Round to bytes after each axis, as in the RGB reference processor.
    static func resizeRGB(
        _ pixels: [UInt8], width: Int, height: Int, targetWidth: Int, targetHeight: Int
    ) -> [UInt8] {
        func coefficients(_ input: Int, _ output: Int) -> [(Int, [Int])] {
            let scale = Double(input) / Double(output)
            let support = max(1, scale)
            return (0 ..< output).map { index in
                let center = (Double(index) + 0.5) * scale
                let first = max(0, Int(center - support + 0.5))
                let last = min(input, Int(center + support + 0.5))
                let weights = (first ..< last).map {
                    max(0, 1 - abs((Double($0) + 0.5 - center) / support))
                }
                let sum = weights.reduce(0, +)
                return (
                    first, weights.map { Int(($0 / sum * Double(1 << 22) + 0.5).rounded(.down)) }
                )
            }
        }
        var horizontal = pixels
        if width != targetWidth {
            let taps = coefficients(width, targetWidth)
            horizontal = [UInt8](repeating: 0, count: targetWidth * height * 3)
            for y in 0 ..< height {
                for x in 0 ..< targetWidth {
                    for channel in 0 ..< 3 {
                        let (start, weights) = taps[x]
                        var sum = 1 << 21
                        for (tap, weight) in weights.enumerated() {
                            sum += Int(pixels[(y * width + start + tap) * 3 + channel]) * weight
                        }
                        horizontal[(y * targetWidth + x) * 3 + channel] = UInt8(clamping: sum >> 22)
                    }
                }
            }
        }
        guard height != targetHeight else { return horizontal }
        let taps = coefficients(height, targetHeight)
        var result = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
        for y in 0 ..< targetHeight {
            for x in 0 ..< targetWidth {
                for channel in 0 ..< 3 {
                    let (start, weights) = taps[y]
                    var sum = 1 << 21
                    for (tap, weight) in weights.enumerated() {
                        sum +=
                            Int(horizontal[((start + tap) * targetWidth + x) * 3 + channel])
                            * weight
                    }
                    result[(y * targetWidth + x) * 3 + channel] = UInt8(clamping: sum >> 22)
                }
            }
        }
        return result
    }

}
