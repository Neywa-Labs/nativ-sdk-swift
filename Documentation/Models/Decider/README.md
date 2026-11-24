# Decider

[Decider-2b](https://huggingface.co/nativ-community/decider-2b) returns calibrated choice, Boolean, and score answers to named questions.

## Swift

```swift
import NativSDK

let model = try await NativDecisionModel.loadModel(
    from: .repository("nativ-community/decider-2b"))
let result = try await model.predict(
    state: "Please refund my duplicate charge",
    questions: [
        "department": .choice("Which team should handle this ticket?", criteria: ["billing", "technical", "sales"]),
        "refund": .bool("Does the customer request a refund?"),
        "urgency": .score("How urgent is this ticket?", levels: ["Low urgency", "Normal urgency", "High urgency"]),
    ]
).result()
print(result.answers["department"]!.value)
```

Choice supports 2–255 unique labels, optionally with descriptions supplied as `DecisionCriterion("label", description: "...")`. Score uses 2–10 ordered level descriptions and returns an expected level index. Boolean answers include the probability of true; `noul` is accepted as a Boolean alias. Multi-label questions are not supported by Decider.

`probabilities` describes an exclusive choice or score distribution. Confidence, certainty, score legends, and isolated-level fit scores live in `metadata`; isolated fit scores are independent and need not sum to one. The checkpoint supplies calibration settings automatically.

## Evaluation options

`DecisionPredictionOptions` defaults to independent questions and a 32768-token context limit. `isolated: nil` uses the checkpoint setting, so this checkpoint evaluates score levels separately. Set `isolated: false` to score levels together, or use `.init(independent: false)` to pack all questions into one prompt. Packed questions can attend to earlier questions and disable score isolation.

Dictionary questions use alphabetical name order. Pass an array of `NamedDecisionQuestion` to control packed order. Criteria retain their array order.

Use `.text(state)` for a String variable or `.json(JSONValue)` for structured
state. Use text if exact JSON formatting matters. `maxStateTokens` limits the
state text while preserving questions and options. Long contexts need more memory.

## CLI

Save the questions as `questions.json`:

```json
{
  "department": {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": ["billing", "technical", "sales"]
  },
  "refund": {
    "type": "bool",
    "instructions": "Does the customer request a refund?"
  }
}
```

```sh
swift run nativ predict --model nativ-community/decider-2b \
  --state "Please refund my duplicate charge" --questions-file questions.json
```

Results are JSON on stdout, with named `answers` and `usage.input_tokens`; `usage.output_tokens` is always zero. Use `--packed`, `--isolated`, `--no-isolated`, or `--max-state-tokens` to override evaluation defaults. `--state-file` reads text from a file; `--json-state` parses it as structured JSON. Question and criterion mappings use alphabetical key order; lists retain their supplied order.

## Checkpoints

Use the prepared `nativ-community/decider-2b` checkpoint or a complete local copy
with `.directory(url)`. Keep its configuration, weights, and tokenizer together.
Older Mapika conversions require Decider metadata before they can be loaded.
See [decision models](../Decisions/README.md) for the shared prediction API.
