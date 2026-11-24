// Copyright © 2025 Apple Inc.

/// Supported tool call formats for different language models.
///
/// This enum defines the various tool call formats used by different LLM families.
/// Each format has its own syntax for encoding function names and arguments.
///
/// The raw string values can be used for JSON serialization or CLI parameters.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public enum ToolCallFormat: String, Hashable, Sendable, Codable, CaseIterable {
    /// Default JSON format used by Llama, Qwen, and most models.
    /// Example: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
    case json

    /// LFM2/LFM2.5 Pythonic format with model-specific tags.
    /// Example: `<|tool_call_start|>[func(arg='value')]<|tool_call_end|>`
    case lfm2

    /// XML function format used by Nemotron, Qwen3 Coder, Qwen3 Next, and similar models.
    /// Example: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    case xmlFunction = "xml_function"

    /// Qwen 3.5's XML function format with a framed Hermes-JSON compatibility dialect.
    ///
    /// Qwen 3.5 is prompted to emit `xmlFunction`, but can sporadically emit the
    /// Qwen/Hermes JSON dialect used by earlier Qwen models instead. Both dialects
    /// use the same `<tool_call>` frame; this format accepts either payload without
    /// enabling bare JSON recovery.
    case qwen35 = "qwen3_5"

    /// GLM4 format with arg_key/arg_value tags.
    /// Example: `func<arg_key>k</arg_key><arg_value>v</arg_value>`
    case glm4

    /// Ling's named XML calls, including functions with no arguments.
    case ling3

    /// Gemma function call format.
    /// Example: `<start_function_call>call:name{key:value,k:<escape>str<escape>}<end_function_call>`
    case gemma

    /// Gemma4 function call format.
    /// Example: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`
    case gemma4

    /// Kimi K2 format with functions prefix.
    /// Example: `functions.name:0<|tool_call_argument_begin|>{"key": "value"}`
    case kimiK2 = "kimi_k2"

    /// MiniMax M2 format with invoke/parameter tags.
    /// Example: `<invoke name="f"><parameter name="k">v</parameter></invoke>`
    case minimaxM2 = "minimax_m2"

    /// Muse Glimmer's Onyx ATEM invoke/parameter format.
    /// Example: `<atem:function_calls><atem:invoke name="f">...</atem:invoke></atem:function_calls>`
    case atem

    /// Inkling 2 assistant frames with JSON tool invocations.
    case inkling2

    /// Mistral V11+ format with [TOOL_CALLS] and [ARGS] delimiters.
    /// Example: `[TOOL_CALLS]get_weather [ARGS]{"location": "Tokyo"}`
    case mistral

    /// Llama 3 inline JSON format.
    /// Example: `<|python_tag|>{ "name": "func", "parameters": {...} }`
    case llama3

    /// GPT-OSS full Harmony response protocol.
    ///
    /// Not a tool-call JSON dialect: selects the Harmony frame parser + router
    /// (channels, recipients, terminators). Tool calls appear only as
    /// `commentary to=functions.<name>` frames within that protocol.
    /// Example: `<|channel|>commentary to=functions.get_weather<|message|>{"location": "Tokyo"}<|call|>`
    case gptOSS = "gpt_oss"

}
