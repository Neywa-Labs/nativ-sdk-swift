import MLX

/// Feature insertion for models using a single sequence of image/video placeholders.
enum MultimodalEmbedding {
    static func mergeInputIDsWithImageFeatures(
        inputIDs: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray,
        imageTokenID: Int, videoTokenID: Int,
    )
        -> MLXArray
    {
        var imageIndices = [Int]()
        for (index, value) in inputIDs.asArray(Int.self).enumerated()
        where value == imageTokenID || value == videoTokenID {
            imageIndices.append(index)
        }

        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }

        if imageFeatures.ndim == 2 {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures[.newAxis, 0..., 0...]
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }
}
