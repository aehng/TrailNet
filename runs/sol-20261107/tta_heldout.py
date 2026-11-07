"""One heldout comparison after frozen VAL selection. No threshold or sibling tuning."""
import sys,json,hashlib,os
from pathlib import Path
import numpy as np,torch,rasterio
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trailnet.models.unet import build_model
from trailnet.config import load_config
from trailnet.data.dataset import TrailTileDataset
from trailnet.frozen_eval import frozen_counts
from trailnet.tta import TTA4,transform_input,invert_output
OUT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    selected=json.loads((OUT/'frozen-selection.json').read_text())
    val=OUT/'val-output/tta-val-results.json';assert digest(val)==selected['val_results_sha256']
    assert selected['winner']=='TTA4' and selected['threshold']==.9 and selected['val_f1']>=selected['val_gate']
    assert torch.cuda.is_available();ckpt=Path(os.environ['SOL_CHECKPOINT']);assert digest(ckpt)==selected['checkpoint_sha256']
    cfg=load_config(ROOT/'config_exp_t4e2_a_ignore5.yaml');cfg['model']['encoder_weights']=None
    model=build_model(cfg['model']).cuda();model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=False)['model']);model.eval()
    rows=json.loads((OUT/'test-rows.json').read_text());assert len(rows)==358
    ds=TrailTileDataset(rows,pixel_weights=cfg['loss']['pixel_weights']);bounds={};masks={}
    source=Path(os.environ['SOL_SOURCE_ROOT'])
    for a in cfg['areas']:
        with rasterio.open(source/a['name']/'naip_rgbn.tif') as src:bounds[a['name']]=src.bounds
    for r in rows:
        with rasterio.open(Path(r['index_dir'])/r['image']) as tile:
            rr,cc=np.indices((512,512));t=tile.transform;xx=t.c+(cc+.5)*t.a;yy=t.f+(rr+.5)*t.e
            l,b,ri,top=bounds[r['area']];masks[(r['area'],r['image'])]=torch.from_numpy((xx<l)|(xx>=ri)|(yy<b)|(yy>=top))[None]
    payload='\n'.join(f'{a}\t{im}\t{hashlib.sha256(m.numpy().tobytes()).hexdigest()}' for (a,im),m in sorted(masks.items()));mask_sha=hashlib.sha256(payload.encode()).hexdigest()
    scores={a:{k:dict(tp=0,fp=0,fn=0) for k in ('legacy','coverage')} for a in ('A','TTA4')}
    by_area={a:{name:{k:dict(tp=0,fp=0,fn=0) for k in ('legacy','coverage')} for name in sorted({r['area'] for r in rows})} for a in scores}
    for i,r in enumerate(rows):
        x,y,w=ds[i];x=x[None].cuda();outputs={};total=None
        with torch.inference_mode():
            for turns,flip in TTA4:
                p=invert_output(model(transform_input(x,turns,flip).contiguous()).sigmoid(),turns,flip).cpu()[0]
                if (turns,flip)==(0,False):outputs['A']=p
                total=p if total is None else total+p
        outputs['TTA4']=total/4
        for a,p in outputs.items():
            for kind,mask in [('legacy',None),('coverage',masks[(r['area'],r['image'])])]:
                c=frozen_counts(p,y,w,.9,mask)
                for key in c:scores[a][kind][key]+=c[key];by_area[a][r['area']][kind][key]+=c[key]
        pdir=Path('/kaggle/working/heldout-probabilities');pdir.mkdir(exist_ok=True);np.savez_compressed(pdir/(r['image']+'.npz'),**{k:v.numpy() for k,v in outputs.items()})
        if i%40==0:print('HELDOUT_PROGRESS',i+1,len(rows),flush=True)
    for a in scores:
        for kind,c in scores[a].items():
            c['f1']=2*c['tp']/(2*c['tp']+c['fp']+c['fn']);c['precision']=c['tp']/max(1,c['tp']+c['fp']);c['recall']=c['tp']/max(1,c['tp']+c['fn'])
            assert c['tp']+c['fn']==scores['A'][kind]['tp']+scores['A'][kind]['fn']
    baseline=scores['A']['legacy'];assert abs(baseline['f1']-.4311)<.0001,baseline
    c=scores['TTA4']['coverage'];passes=c['f1']>=scores['A']['coverage']['f1'] and c['precision']>=.53 and c['recall']>=.33 and scores['TTA4']['legacy']['f1']>=.4311
    result=dict(status='complete',selection=selected,heldout_scored=True,heldout_candidate_count=1,heldout_tile_count=358,threshold=.9,training_launched=False,coverage_mask_sha256=mask_sha,scores=scores,per_area=by_area,heldout_veto_passed=passes,independent_truth_verified=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    Path('/kaggle/working/tta-heldout-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HELDOUT_RESULT',json.dumps(scores),'VETO_PASS',passes,flush=True)
if __name__=='__main__':main()
