"""Regenerate the tiny float32 parity fixture with the mlx-vlm revision in provenance.json.

Usage: python generate.py /path/to/Fixtures/LFM3
"""

import json, sys, pathlib
import mlx.core as mx, numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.lfm3_omni.config import ModelConfig
from mlx_vlm.models.lfm3_omni.lfm3_omni import Model
from mlx_vlm.models.lfm3_audio.audio import log_mel
from mlx_vlm.models.lfm3_audio.depth import sample_frame
root = pathlib.Path(sys.argv[1])
root.mkdir(exist_ok=True, parents=True)
text = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=96, norm_eps=1e-05, conv_L_cache=3, layer_types=['conv', 'full_attention', 'sliding_attention'], sliding_window=4, rope_parameters={'full_attention': {'rope_theta': 10000, 'partial_rotary_factor': 0.5}, 'sliding_attention': {'rope_theta': 1000}}, use_sandwich_norm=True)
audio_text = dict(model_type='lfm2', max_position_embeddings=1024, conv_bias=False, block_ffn_dim_multiplier=1.0, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, vocab_size=2, norm_eps=1e-05, conv_L_cache=3, layer_types=['conv', 'full_attention'], block_auto_adjust_ff_dim=False, block_multiple_of=8)
depth = {**text, 'num_hidden_layers': 2, 'layer_types': ['full_attention'] * 2, 'use_sandwich_norm': False}
vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, in_channels=3, patch_size=2, temporal_patch_size=2, spatial_merge_size=2, layer_types=['conv', 'full_attention'])
config = dict(model_type='lfm3_omni', text_config=text, vision_config=vision, audio_encoder_config=dict(text_config=audio_text, conv_channels=4, hidden_size=16, num_mel_bins=128, sliding_window=4), depth_config=depth, num_codebooks=3, codebook_size=9, audio_eos_code=8, maskgit_schedule=[1, 2], maskgit_steps=2, interleaved_n_text=2, interleaved_n_audio=3, audio_token_id=80, audio_out_token_id=81, audio_start_token_id=82, text_end_token_id=83, mixed_start_token_id=84, image_token_id=85, video_token_id=86, projector_hidden_size=32, downsample_factor=2)
(root / 'config.json').write_text(json.dumps(config, indent=2))
model = Model(ModelConfig.from_dict(config))
mx.random.seed(178)
weights = {}
for (k, v) in tree_flatten(model.parameters()):
    if 'norm' in k and k.endswith('weight'):
        weights[k] = mx.ones(v.shape)
    elif k.endswith('bias'):
        weights[k] = mx.zeros(v.shape)
    else:
        weights[k] = mx.random.normal(v.shape) * 0.08
model.load_weights(list(weights.items()))
model.eval()
mx.save_safetensors(str(root / 'weights.safetensors'), weights)
ids = mx.array([[1] + [85] * 4 + [80] * 5 + [2, 3]])
features = mx.sin(mx.arange(2 * 128 * 17).reshape(2, 128, 17) * 0.07)
mask = mx.arange(17)[None] < mx.array([17, 9])[:, None]
pixels = mx.cos(mx.arange(16 * 24).reshape(16, 24) * 0.04)
kwargs = dict(input_features=features, feature_attention_mask=mask, image_grid_thw=mx.array([[1, 4, 4]]))
embed = model.get_input_embeddings(ids, pixels, **kwargs).inputs_embeds
h = model.language_model.model(None, input_embeddings=embed)
logits = model.language_model(ids, inputs_embeds=embed).logits
hidden = mx.arange(16, dtype=mx.float32).reshape(1, 16) / 16
codes = mx.array([[1, 2, 3]])
masked = mx.array([[True, False, True]])
wave = mx.sin(mx.arange(1900) * 0.07) * 0.25
result = dict(ids=ids, features=features, mask=mask, pixels=pixels, embeddings=embed, hidden=h, logits=logits, depth_hidden=hidden, depth_codes=codes, depth_masked=masked, depth_logits=model.depth_decoder(hidden, codes, masked), depth_frame=sample_frame(model.depth_decoder, hidden, [1, 2], temperature=0, order=mx.array([[2, 0, 1]])), wave=wave, log_mel=mx.array(log_mel(np.array(wave))))
mx.save_safetensors(str(root / 'golden.safetensors'), result)
provenance = json.loads(pathlib.Path(__file__).with_name('provenance.json').read_text())
provenance['mlx_version'] = mx.__version__
(root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print({k: v.shape for (k, v) in result.items()})
