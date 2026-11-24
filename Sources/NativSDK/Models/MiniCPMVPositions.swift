import Foundation
import MLX

/// Canvas M-RoPE from Transformers' MiniCPMV4_7Model.get_rope_index. Crops use the
/// processor's explicit bounds, so marker-looking text cannot consume an image grid.
enum MiniCPMVCanvasPositions {
    struct Frame {
        let thumbnail: Int
        var slices: [Int] = []
        var end = 0
    }

    static func make(
        tokens: [Int], bounds: [Range<Int>], grids: [THW], configuration: MiniCPMVConfiguration
    ) throws -> (positions: MLXArray, delta: Int) {
        let factor = configuration.downsampleMode == "4x" ? 2 : 4
        let markers = Set([
            configuration.sliceStartID, configuration.sliceEndID,
            configuration.imageStartID, configuration.imageEndID, configuration.newlineID,
        ])
        guard bounds.count == grids.count else {
            throw VLMError.processing("MiniCPM-V crop/grid count mismatch")
        }
        var previousEnd = 0
        var groups: [[Frame]] = []
        for (index, bound) in bounds.enumerated() {
            let grid = grids[index]
            guard bound.lowerBound > 0, bound.upperBound < tokens.count,
                bound.lowerBound >= previousEnd, grid.h > 0, grid.w > 0,
                grid.h % factor == 0, grid.w % factor == 0,
                bound.count == (grid.h / factor) * (grid.w / factor)
            else { throw VLMError.processing("MiniCPM-V invalid visual crop bounds or dimensions") }
            if !groups.isEmpty, tokens[bound.lowerBound - 1] == configuration.sliceStartID {
                groups[groups.count - 1][groups[groups.count - 1].count - 1].slices.append(index)
            } else {
                let adjacent =
                    !groups.isEmpty
                    && tokens[previousEnd ..< bound.lowerBound - 1].allSatisfy {
                        markers.contains($0)
                    }
                if adjacent {
                    groups[groups.count - 1].append(Frame(thumbnail: index))
                } else {
                    groups.append([Frame(thumbnail: index)])
                }
            }
            previousEnd = bound.upperBound
        }
        for group in groups.indices {
            for frame in groups[group].indices {
                let item = groups[group][frame]
                var end = bounds[item.slices.last ?? item.thumbnail].upperBound
                while end < tokens.count, markers.contains(tokens[end]) { end += 1 }
                groups[group][frame].end = end
            }
        }
        var positions = Array(repeating: Array(repeating: 0, count: tokens.count), count: 3)
        func text(_ range: Range<Int>, start: Int) {
            for i in range {
                for channel in 0 ..< 3 { positions[channel][i] = start + i - range.lowerBound }
            }
        }
        func crop(
            _ index: Int, origin: Int, canvasH: Int = 0, canvasW: Int = 0, hOffset: Int = 0,
            wOffset: Int = 0
        ) {
            let h = grids[index].h / factor
            let w = grids[index].w / factor
            let height = canvasH == 0 ? h : canvasH
            let width = canvasW == 0 ? w : canvasW
            for row in 0 ..< h {
                for column in 0 ..< w {
                    let token = bounds[index].lowerBound + row * w + column
                    positions[0][token] = origin
                    positions[1][token] =
                        origin + hOffset
                        + (h == 1
                            ? 0
                            : Int(
                                (Double(row * (height - 1)) / Double(h - 1)).rounded(
                                    .toNearestOrEven)))
                    positions[2][token] =
                        origin + wOffset
                        + (w == 1
                            ? 0
                            : Int(
                                (Double(column * (width - 1)) / Double(w - 1)).rounded(
                                    .toNearestOrEven)))
                }
            }
        }
        var currentPosition = 0
        var cursor = 0
        for group in groups {
            let start = bounds[group[0].thumbnail].lowerBound - 1
            let end = group.last!.end
            text(cursor ..< start, start: currentPosition)
            currentPosition += start - cursor
            var frameCursor = start
            for frame in group {
                let thumb = bounds[frame.thumbnail]
                let frameStart = thumb.lowerBound - 1
                if frameStart > frameCursor {
                    text(frameCursor ..< frameStart, start: currentPosition)
                    currentPosition += frameStart - frameCursor + 1
                }
                let frameEnd = min(frame.end, end)
                let origin = currentPosition
                var height = grids[frame.thumbnail].h / factor
                var width = grids[frame.thumbnail].w / factor
                var columns = frame.slices.count
                var sliceH = 0
                var sliceW = 0
                if let first = frame.slices.first {
                    for k in 0 ..< max(0, frame.slices.count - 1) {
                        if bounds[frame.slices[k + 1]].lowerBound
                            - bounds[frame.slices[k]].upperBound > 2
                        {
                            columns = k + 1
                            break
                        }
                    }
                    if frame.slices.count % columns != 0 { columns = frame.slices.count }
                    sliceH = grids[first].h / factor
                    sliceW = grids[first].w / factor
                    height = (frame.slices.count / columns) * sliceH
                    width = columns * sliceW
                }
                for i in frameStart ..< frameEnd {
                    for channel in 0 ..< 3 { positions[channel][i] = origin }
                }
                positions[1][frameStart] = max(origin - 1, 0)
                positions[2][frameStart] = max(origin - 1, 0)
                if thumb.upperBound < frameEnd {
                    positions[1][thumb.upperBound] = origin + height
                    positions[2][thumb.upperBound] = origin + width
                }
                crop(frame.thumbnail, origin: origin, canvasH: height, canvasW: width)
                for (k, index) in frame.slices.enumerated() {
                    let bound = bounds[index]
                    let hOffset = (k / columns) * sliceH
                    let wOffset = (k % columns) * sliceW
                    positions[1][bound.lowerBound - 1] = origin + hOffset
                    positions[2][bound.lowerBound - 1] = origin + wOffset
                    if bound.upperBound < frameEnd {
                        positions[1][bound.upperBound] =
                            origin + hOffset + grids[index].h / factor - 1
                        positions[2][bound.upperBound] =
                            origin + wOffset + grids[index].w / factor - 1
                    }
                    crop(index, origin: origin, hOffset: hOffset, wOffset: wOffset)
                    if k + 1 < frame.slices.count {
                        let next = bounds[frame.slices[k + 1]].lowerBound
                        if next - bound.upperBound > 2 {
                            for i in bound.upperBound + 1 ..< next - 1 {
                                positions[1][i] = origin + (k / columns + 1) * sliceH - 1
                                positions[2][i] = origin + columns * sliceW
                            }
                        }
                    }
                }
                currentPosition = origin + max(height, width) + 1
                frameCursor = frameEnd
            }
            if frameCursor < end {
                text(frameCursor ..< end, start: currentPosition)
                currentPosition += end - frameCursor
            }
            cursor = end
        }
        text(cursor ..< tokens.count, start: currentPosition)
        let flat = positions.flatMap { $0 }
        return (MLXArray(flat, [3, 1, tokens.count]), (flat.max() ?? -1) + 1 - tokens.count)
    }
}
