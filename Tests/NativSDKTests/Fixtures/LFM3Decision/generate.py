"""Generate d1 set-attention fixtures from mlx-vlm, reusing the tiny LFM3 weights.
Run inside the reference mlx-vlm environment with this folder as the argument.
"""
import json, pathlib, sys
import mlx.core as mx
from mlx_vlm.models.d1_lfm3_vl.d1 import set_masks
from mlx_vlm.models.d1_lfm3_omni import Model, ModelConfig
mx.set_default_device(mx.cpu)
root=pathlib.Path(sys.argv[1]); base=root.parent/'LFM3'
c=json.loads((base/'config.json').read_text());c.update(architectures=['D1Model'],d1_attention='set',d1_set_size=16,d1_codes=2,d1_numbers=True)
(root/'config.json').write_text(json.dumps(c,indent=2)+'\n')
m=Model(ModelConfig.from_dict(c));m.load_weights(list(mx.load(str(base/'weights.safetensors')).items()));m.eval()
g=mx.load(str(base/'golden.safetensors'))
# Use one audio item and one image, two question groups, across the sliding window.
media=[m.d1_features(dict(type='image',pixel_values=g['pixels'],image_grid_thw=mx.array([[1,4,4]]))),m.d1_features(dict(type='audio',input_features=g['features'][:1],feature_attention_mask=g['mask'][:1]))]
n=media[1][1].shape[0]
ids=[1]+[85]*4+[80]*n+[2,3,4,5,6,7]
state=1+4+n;groups=[-1]*state+[1]*3+[2]*3
h=m.d1_hidden(ids,state,groups,media)
mx.save_safetensors(str(root/'golden.safetensors'),dict(ids=mx.array(ids),groups=mx.array(groups),state=mx.array(state),hidden=h,full_mask=set_masks(len(ids),state,groups,4)['full_attention'],image_features=media[0][1],audio_features=media[1][1]))
(root/'provenance.json').write_text(json.dumps(dict(reference='mlx-vlm',revision='4693cd0eaf022ae7ef61c6fdb55e17e6399879a7',mlx_version=mx.__version__,weights='../LFM3/weights.safetensors',device='cpu',license='LFM Open License v1.0'),indent=2)+'\n')
print(h.shape)
