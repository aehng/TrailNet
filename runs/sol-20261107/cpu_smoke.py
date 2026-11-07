import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trailnet.models.unet import build_model
from trailnet.config import load_config
from trailnet.tta import ensemble_probabilities,TTA4,TTA8
from trailnet.data.dataset import TrailTileDataset
cfg=load_config(ROOT/'config_exp_t4e2_a_ignore5.yaml');cfg['model']['encoder_weights']=None
torch.set_num_threads(4)
model=build_model(cfg['model']);model.load_state_dict(torch.load('/workspace/kaggle-t4e/ckpt-a/trailnet_best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
row=json.loads((Path(__file__).parent/'val-rows.json').read_text())[0]
x,y,w=TrailTileDataset([row],pixel_weights=cfg['loss']['pixel_weights'])[0]
with torch.inference_mode():
    for transforms in (TTA4,TTA8):
        p=ensemble_probabilities(model,x[None,:,:64,:64],transforms)
        assert p.shape==(1,1,64,64) and p.isfinite().all() and p.min()>=0 and p.max()<=1
        print('CPU_SMOKE_OK',len(transforms),float(p.min()),float(p.max()),flush=True)
