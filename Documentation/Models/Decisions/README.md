# Decision models

Load decision models with `NativDecisionModel.loadModel` and call `predict` to obtain named, typed answers without generating text.

```swift
import NativSDK

let model = try await NativDecisionModel.loadModel(
    from: .repository("nativ-community/decider-2b"))
let result = try await model.predict(
    state: "Please refund my duplicate charge",
    questions: ["refund": .bool("Does the customer request a refund?")]
).result()
print(result.answers["refund"]!.value)
```

Keep the prediction handle to cancel a request with `cancel()`, and await
`result()` for the answers.

Questions use `.choice(instructions, criteria:)`, `.score(instructions, levels:)`, or `.bool(instructions)`. Support is model-specific; unsupported question types fail before inference. Answer values are `.choice(String)`, `.score(Double)`, or `.bool(Bool)`. Exclusive distributions are in `probabilities`, Boolean true probability is in `probability`, and model-specific metrics remain in `metadata`.

`DecisionPredictionOptions` controls evaluation modes and the state token limit; supported modes are model-specific. Decider supports packed evaluation and optional score isolation. Laya evaluates questions independently and does not accept packed or isolation overrides. Dictionary questions have alphabetical order; an array of `NamedDecisionQuestion` preserves explicit order. Structured state can be passed as `.json(JSONValue)`; String variables use `.text(string)`.

Supported models are [Decider-2b](../Decider/README.md), [Laya](../Laya/README.md), and [LFM3 d1 / d1-Omni](../LFM3/README.md#d1-decision-models). Their guides cover answer probabilities, criteria, evaluation options, and the `nativ predict` CLI. Confidence metrics in `metadata` retain each model's own meaning.
