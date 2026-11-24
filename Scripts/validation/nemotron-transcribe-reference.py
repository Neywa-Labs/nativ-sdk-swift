"""Create local differential fixtures with the MLX Audio reference implementation.

Run in an environment containing mlx-audio at commit
92decb7577878ef8e9274c1a7e33084760be03aa. No downloads are performed.
The audio directory must contain announcement.wav, conversation.wav,
spoken_request.wav and rich_speech.wav; all arrays are resampled to mono 16 kHz.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.io import wavfile
from transformers import PreTrainedTokenizerFast

from mlx_audio.stt.models.nemotron_transcribe.config import ModelConfig
from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import resolve_prompt
from mlx_audio.stt.models.nemotron_transcribe.perception import Perception, normalize_features
from mlx_audio.stt.utils import load_audio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("audio_directory", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
config = ModelConfig.from_dict(json.loads((args.source / "config.json").read_text()))
index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
weights = {}
for shard in sorted({v for k, v in index.items() if k.startswith("perception.")}):
    weights.update({k.removeprefix("perception."): v
                    for k, v in mx.load(str(args.source / shard)).items()
                    if k.startswith("perception.")})
model = Perception(config)
model.load_weights(list(weights.items()), strict=True)
mx.eval(model.parameters())
mx.save_safetensors(str(args.output / "perception.safetensors"), weights)
for name in ["announcement", "conversation", "spoken_request", "rich_speech", "long"]:
    audio = load_audio(str(args.audio_directory / ("conversation.wav" if name == "long" else name + ".wav")), sr=16000)
    if name == "long":
        audio = mx.tile(audio, [5])
    wavfile.write(str(args.output / (name + ".wav")), 16000, np.array(audio).astype("float32"))
    mel = model.preprocessor(audio)
    normalized = normalize_features(mel)
    encoded = model.encoder(mel)
    projected = model.proj(encoded)
    mx.eval(projected)
    mx.save_safetensors(str(args.output / (name + ".safetensors")),
                        dict(audio=audio, mel=mel, normalized=normalized, encoded=encoded, projected=projected))
    print(name, audio.size / 16000, projected.shape, flush=True)

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.source, fix_mistral_regex=False)
records = []
for task, language, prompt in [("asr", None, None), ("translate", "German", None),
                               ("saa", None, None), ("qa", None, "What time is the design review?")]:
    prompt = resolve_prompt(task, prompt, language, None, "<|audio|>")
    turns = ["<|im_start|>system\n<|im_end|>\n", f"<|im_start|>user\n{prompt}<|im_end|>\n",
             "<|im_start|>assistant\n<think></think>"]
    ids = sum([tokenizer.encode(turn, add_special_tokens=False) for turn in turns], [])
    records.append(dict(task=task, language=language, prompt=prompt, ids=ids))
(args.output / "prompts.json").write_text(json.dumps(records, indent=2))
