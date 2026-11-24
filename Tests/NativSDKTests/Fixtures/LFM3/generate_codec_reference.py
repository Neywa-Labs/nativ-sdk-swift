"""Generate a real LiquidCodec reference using the mlx-vlm revision in provenance.json.

Usage: python generate_codec_reference.py /path/to/model /tmp/codec.safetensors
"""

import argparse

import mlx.core as mx
from mlx_vlm.models.lfm3_audio.codec import load_codec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
parser.add_argument("output")
args = parser.parse_args()

codec = load_codec(args.checkpoint, dtype=mx.float32)
codes = (mx.arange(codec.config.n_q * 32).reshape(codec.config.n_q, 32) * 137) % codec.config.codebook_size
mx.save_safetensors(args.output, {"codes": codes, "waveform": codec.decode(codes)})
