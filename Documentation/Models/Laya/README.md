# Laya

[Laya](https://huggingface.co/nativ-community/laya) answers named choice, Boolean, and score questions without generating a text response.

```swift
import NativSDK

let model = try await NativDecisionModel.loadModel(
    from: .repository("nativ-community/laya"))
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

Choice labels may have descriptions: `DecisionCriterion("billing", description: "Payments and refunds")`. Scores use ordered level descriptions and return an expected level index, rounded to four decimal places. Boolean answers include the probability of true and use 0.5 as the decision threshold; `noul` is accepted as an alias. Multi-label decisions are unsupported.

## Calibration and metadata

The checkpoint supplies calibration settings automatically.

Choice and score answers expose exclusive `probabilities` and entropy confidence in `metadata["confidence"]`. This confidence differs from Decider's maximum-option confidence. Additional action metadata is available in `metadata["rl_agent"]["act_probability"]`. This is not an answer probability. Score metadata also includes the level legend.

## Inputs and limits

Keep question instructions and option descriptions concise. The checkpoint limits
input length and may shorten state, instructions, and descriptions to fit.
`maxStateTokens` can further limit the state text.

Use `.text(state)` for String variables or `.json(JSONValue)` for structured state.
Use text to preserve an existing JSON string's formatting.

Questions are evaluated independently. Packed evaluation and explicit isolation
overrides are unsupported. `usage.input_tokens` reports input tokens across the
questions; `usage.output_tokens` is zero.

## CLI

Use the [shared question JSON format](../Decider/README.md#cli):

```sh
swift run nativ predict --model nativ-community/laya \
  --state "Please refund my duplicate charge" --questions-file questions.json
```

`--state-file`, `--json-state`, `--revision`, and `--max-state-tokens` are supported. Leave `--packed`, `--isolated`, and `--no-isolated` unset for Laya.

## Checkpoints

Use `nativ-community/laya` or a complete local copy with `.directory(url)`. Keep the
matching configuration, weights, and tokenizer together.

See [decision models](../Decisions/README.md) for prediction and cancellation.
