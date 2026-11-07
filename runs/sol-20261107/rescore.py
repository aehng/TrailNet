"""CPU VAL-only replay of existing candidate against ORIGINAL A supervision."""
import sys,json,hashlib,time,argparse
from pathlib import Path
import torch, numpy as np, rasterio
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from trailnet.data.dataset import TrailTileDataset
from trailnet.models.unet import build_model
from trailnet.config import load_config
from trailnet.frozen_eval import frozen_counts
OUT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=['prior','A'],default='prior'); args=parser.parse_args()
    torch.set_num_threads(4)
    cfg=load_config(ROOT/'config_exp_t4e2_a_ignore5.yaml'); cfg['model']['encoder_weights']=None
    rows=json.loads((OUT/'val-rows.json').read_text()); assert len(rows)==172
    ds=TrailTileDataset(rows,pixel_weights=cfg['loss']['pixel_weights'])
    model=build_model(cfg['model']); ckpt=OUT/'prior-candidate/coverage-ignore/trailnet_best.pt'
    if args.arm=='A': ckpt=Path('/workspace/kaggle-t4e/ckpt-a/trailnet_best.pt')
    expected='73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a' if args.arm=='A' else 'efc92f7ec027d006f3854865c414746396ce95379c2afd371cbdcb81c1664699'
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==expected
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model']); model.eval()
    result={name:dict(tp=0,fp=0,fn=0) for name in ('legacy','coverage')}; masks={}
    for a in cfg['areas']:
        with rasterio.open(Path('/workspace/trailnet-eval-vb-check/TrailNet/data')/a['name']/'naip_rgbn.tif') as source: masks[a['name']]=source.bounds
    with torch.inference_mode():
        for i,r in enumerate(rows):
            x,y,w=ds[i]
            with rasterio.open(Path(r['index_dir'])/r['image']) as tile:
                rr,cc=np.indices((512,512)); t=tile.transform
                xx=t.c+(cc+.5)*t.a; yy=t.f+(rr+.5)*t.e
                l,b,ri,top=masks[r['area']]; outside=torch.from_numpy((xx<l)|(xx>=ri)|(yy<b)|(yy>=top))[None]
            prob=model(x[None]).sigmoid()[0]
            for name,mask in [('legacy',None),('coverage',outside)]:
                c=frozen_counts(prob,y,w,.9,mask)
                for k in c: result[name][k]+=c[k]
            # Retain predictions for reproducibility without inference replay.
            folder=OUT/(args.arm+'-val-probabilities'); folder.mkdir(exist_ok=True)
            np.save(folder/(r['image']+'.npy'),prob.numpy())
            if i%20==0: print(i,result,flush=True)
    for v in result.values(): v['f1']=2*v['tp']/(2*v['tp']+v['fp']+v['fn'])
    assert result['legacy']['tp']+result['legacy']['fn']==177079
    assert result['coverage']['tp']+result['coverage']['fn']==164884
    (OUT/('corrected-'+args.arm+'-rescore.json')).write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),threshold=.9,heldout_scored=False,results=result),indent=2)+'\n')
    print(result,flush=True)
if __name__=='__main__': main()
