"""Frozen VAL comparison. No training, no heldout reads, no adaptive arm choices."""
import sys,json,hashlib,os,time
from pathlib import Path
import numpy as np,torch,rasterio
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trailnet.models.unet import build_model
from trailnet.config import load_config
from trailnet.data.dataset import TrailTileDataset
from trailnet.frozen_eval import frozen_counts
from trailnet.tta import TTA4,TTA8,transform_input,invert_output
OUT=Path(__file__).resolve().parent

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    assert torch.cuda.is_available()
    cfg=load_config(ROOT/'config_exp_t4e2_a_ignore5.yaml');cfg['model']['encoder_weights']=None
    ckpt=Path(os.environ['SOL_CHECKPOINT']); assert digest(ckpt)=='73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a'
    model=build_model(cfg['model']).cuda(); model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model']); model.eval()
    rows=json.loads((OUT/'val-rows.json').read_text()); assert len(rows)==172
    ds=TrailTileDataset(rows,pixel_weights=cfg['loss']['pixel_weights']); source=Path(os.environ['SOL_SOURCE_ROOT'])
    masks={};bounds={}
    for a in cfg['areas']:
        with rasterio.open(source/a['name']/'naip_rgbn.tif') as src: bounds[a['name']]=src.bounds
    for r in rows:
        with rasterio.open(Path(r['index_dir'])/r['image']) as tile:
            t=tile.transform; rr,cc=np.indices((512,512));xx=t.c+(cc+.5)*t.a;yy=t.f+(rr+.5)*t.e
            l,b,ri,top=bounds[r['area']]; masks[(r['area'],r['image'])]=torch.from_numpy((xx<l)|(xx>=ri)|(yy<b)|(yy>=top))[None]
    payload='\n'.join(f'{a}\t{im}\t{hashlib.sha256(m.numpy().tobytes()).hexdigest()}' for (a,im),m in sorted(masks.items()))
    mask_sha=hashlib.sha256(payload.encode()).hexdigest(); assert mask_sha=='6350928231908df0e7092003486735ecb697721a1d6904b026ba4ed7be9bfc52'
    thresholds=[float(v) for v in cfg['training']['val_thresholds']];assert len(thresholds)==14
    scores={arm:{kind:{str(t):dict(tp=0,fp=0,fn=0) for t in thresholds} for kind in ('legacy','coverage')} for arm in ('A','TTA4','TTA8')}
    per_area={arm:{a:{kind:dict(tp=0,fp=0,fn=0) for kind in ('legacy','coverage')} for a in sorted({r['area'] for r in rows})} for arm in scores}
    for i,r in enumerate(rows):
        x,y,w=ds[i]; x=x[None].cuda(); outputs={}; total4=None;total8=None
        with torch.inference_mode():
            for k,f in TTA8:
                p=invert_output(model(transform_input(x,k,f).contiguous()).sigmoid(),k,f).cpu()[0]
                if (k,f)==(0,False): outputs['A']=p
                total8=p if total8 is None else total8+p
                if (k,f) in TTA4: total4=p if total4 is None else total4+p
        outputs.update(TTA4=total4/4,TTA8=total8/8)
        for arm,p in outputs.items():
            for kind,mask in [('legacy',None),('coverage',masks[(r['area'],r['image'])])]:
                for t in thresholds:
                    c=frozen_counts(p,y,w,t,mask)
                    for key in c: scores[arm][kind][str(t)][key]+=c[key]
                    if t==.9:
                        for key in c: per_area[arm][r['area']][kind][key]+=c[key]
        prob_dir=Path('/kaggle/working/val-probabilities');prob_dir.mkdir(exist_ok=True)
        np.savez_compressed(prob_dir/(r['image']+'.npz'),**{k:v.numpy() for k,v in outputs.items()})
        if i%20==0: print('VAL_PROGRESS',i+1,len(rows),flush=True)
    for arm in scores:
        for kind in scores[arm]:
            for c in scores[arm][kind].values():
                assert c['tp']+c['fn']==(177079 if kind=='legacy' else 164884)
                c['f1']=2*c['tp']/(2*c['tp']+c['fp']+c['fn']);c['precision']=c['tp']/max(1,c['tp']+c['fp']);c['recall']=c['tp']/max(1,c['tp']+c['fn'])
    baseline=scores['A']['coverage']['0.9']['f1'];assert abs(baseline-.3536541461658466)<.0001,baseline
    gate=.3736541461658466
    best={arm:max(scores[arm]['coverage'],key=lambda t:(scores[arm]['coverage'][t]['f1'],float(t))) for arm in scores}
    qualified=[a for a in ('TTA4','TTA8') if scores[a]['coverage'][best[a]]['f1']>=gate]
    winner=max(qualified,key=lambda a:(scores[a]['coverage'][best[a]]['f1'],-int(a[3:]))) if qualified else None
    result=dict(status='complete',training_launched=False,heldout_scored=False,checkpoint_sha256=digest(ckpt),coverage_mask_sha256=mask_sha,val_count=172,threshold_grid=thresholds,gate=gate,best_threshold=best,scores=scores,per_area_at_0_90=per_area,winner=winner,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    Path('/kaggle/working/tta-val-results.json').write_text(json.dumps(result,indent=2)+'\n')
    for a in scores: print('RESULT',a,best[a],scores[a]['coverage'][best[a]],flush=True)
    print('FROZEN_WINNER',winner,flush=True)
if __name__=='__main__':main()
