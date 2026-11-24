import Foundation

/// The OCR stage uses Qwen3.5 with a different vocabulary and media-token defaults.
enum IndicOCR {
    static func configuration(from data: Data) throws -> Qwen35Configuration {
        var config = try JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        if config["layout_config"] != nil || config["ocr_config"] != nil {
            throw VLMError.processing(
                "Combined IndicOCR layout/OCR checkpoints are not supported. Load the OCR stage separately."
            )
        }
        if config["text_config"] == nil || config["text_config"] == .null
            || config["vision_config"] == nil || config["vision_config"] == .null
        {
            throw VLMError.processing(
                "IndicOCR requires an OCR-stage checkpoint with text_config and vision_config.")
        }
        for (key, value) in [
            "image_token_id": 262155, "video_token_id": 262156,
            "vision_start_token_id": 262153, "vision_end_token_id": 262154, "vocab_size": 262157,
        ] {
            if config[key] == nil || config[key] == .null { config[key] = .int(value) }
        }
        return try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: JSONEncoder().encode(config))
    }
}
