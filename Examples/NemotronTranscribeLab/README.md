# Nemotron Audio for iPhone and iPad

A SwiftUI example for Nemotron 3.5 Transcribe 30B A4B. Transcribe or translate a
recording, get a speaker-tagged transcript, or ask questions about the audio.
Import a file, record with the microphone, or use the bundled samples. Text
streams after the app processes the full recording. Speaker tags distinguish
voices within that recording. Word timestamps are unavailable.

## Model

Convert the `nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-bf16` checkpoint
on a Mac:

```sh
swift run -c release nativ convert /path/to/bf16 /path/to/4bit --bits 4 --group-size 64
```

Use Xcode's device container tools or `devicectl` to copy the complete converted
checkpoint into the app's `Documents/NemotronTranscribe4bit` directory. Allow
about 19.26 GB of storage. Launch with `--transfer` to keep the device awake
during the copy.

For a smaller checkpoint, add `--quant-predicate mixed` to the conversion
command and copy the result to `Documents/NemotronTranscribe3bitExperts`
(about 18.30 GB). Use the **Weights** picker to choose a checkpoint, then tap
**Load model**.

## Build and run

```sh
xcodegen generate
xcodebuild -project NemotronTranscribeLab.xcodeproj -scheme NemotronTranscribeLab \
  -configuration Release -destination 'generic/platform=iOS' \
  DEVELOPMENT_TEAM=YOUR_TEAM -allowProvisioningUpdates build
```

The app requires the increased-memory-limit entitlement. Inference runs on the
device, and the app saves imported files and recordings in its Documents
directory. Tap **Unload model** to release the model when idle.

## Try the bundled samples

Tap **Test all four tasks** to transcribe the announcement, translate it to German,
answer a question about it, and label the speakers in the conversation. The app
displays the output and reports timing and memory use. It saves the results to
`Documents/results.json`.

## Bundled audio provenance

The WAV files were generated with `gradium/phonon-31m-en` using MLX Audio and resampled to mono 16 kHz. They are synthetic test speech.

- **announcement.wav:** “Tomorrow's design review starts at ten thirty in room four. Please bring the updated prototype.”
- **conversation.wav:** Two synthetic voices. The first speaks the announcement; the second says “I will bring the prototype and the latest test results.”
- **spoken_request.wav:** “What is twelve multiplied by seven? Answer with just the number.” Audio Q&A accepts an empty text prompt, but instruction-following still depends on the model; an explicit question is useful when the audio is source material.

Phonon revision: `92741c5b07d085ff840e26fea8ce43110b87852f`. Voice assets: `NbpkqMVS3CJeq2j8` and `LMDIYE0Oa377Qh6Z` from `gradium/phonon-voices`, revision `0ee17205f254b5abc43b266c8a0f215565c352f7`.
