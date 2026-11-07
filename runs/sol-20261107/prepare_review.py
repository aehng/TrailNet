"""Rebuild original VAL tensors and TRAIN sampler counts; never score heldout."""
import sys, re, json, hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trailnet.config import load_config
from trailnet.data.dataset import read_indexes, spatial_split
from trailnet.data.tiles import _load_label_index, _write_tile
OUT=Path(__file__).resolve().parent
SOURCE=Path('/workspace/trailnet-eval-vb-check/TrailNet/data')
INDEX=Path('/workspace/TrailNet-handoff-20261106-1650/assets/recovered_A/data/tiles_t4e_a')
cfg=load_config(ROOT/'config_exp_t4e2_a_ignore5.yaml')
rows=read_indexes([INDEX/a['name']/'index.csv' for a in cfg['areas']])
split=spatial_split(rows,2500,.15,.15,42)
assert {k:len(v) for k,v in split.items()}==dict(train=1745,val=172,test=358)
records=[]; val_rows=[]; changed=[]
for a in cfg['areas']:
    name=a['name']; folder=OUT/'original-val'/name; folder.mkdir(parents=True,exist_ok=True)
    with rasterio.open(SOURCE/name/'naip_rgbn.tif') as im, rasterio.open(SOURCE/name/'3dep_dtm.tif') as dtm:
        geoms,codes,idx=_load_label_index(Path('/workspace/kaggle-t4e/code/label_overrides')/name/'trails.gpkg',im.crs,cfg['tiling']['trail_buffer_m'],5.,2.)
        profile=im.profile.copy()
        for key in ('blockxsize','blockysize','tiled','interleave'): profile.pop(key,None)
        spec=dict(size=512,relief_radius=25,channel_gain=1.85,out_dir=str(folder),area_name=name,profile=profile,crs=str(im.crs),soft_label_band_m=0,ignore_outside_imagery=False)
        for r in split['train']:
            if r['area']!=name: continue
            rr,cc=map(int,re.search(r'_r(\d+)_c(\d+)',r['image']).groups())
            win=Window(cc,rr,512,512); transform=window_transform(win,im.transform)
            region=box(*rasterio.windows.bounds(win,im.transform)); local={c:[] for c in (4,2,3,1)}
            for i in idx.query(region):
                if geoms[i].intersects(region): local[codes[i]].append((geoms[i],codes[i]))
            shapes=[x for c in (4,2,3,1) for x in local[c]]
            classes=rasterize(shapes,out_shape=(512,512),transform=transform,fill=0,dtype='uint8',all_touched=True) if shapes else np.zeros((512,512),np.uint8)
            positive=np.isin(classes,(1,2,3)); original=int(positive.sum())
            inside=((rr+np.arange(512))[:,None]<im.height)&((cc+np.arange(512))[None,:]<im.width)
            treated=int((positive & inside).sum())
            if bool(original)!=bool(treated): changed.append(dict(area=name,image=r['image'],original_positive=original,treated_positive=treated))
            records.append(dict(area=name,image=r['image'],positive_count=original))
        for r in split['val']:
            if r['area']!=name: continue
            rr,cc=map(int,re.search(r'_r(\d+)_c(\d+)',r['image']).groups())
            if not (folder/r['image']).exists():
                generated=_write_tile(im,dtm,rr,cc,spec,geoms,codes,idx)
                assert all(str(generated[k])==str(r[k]) for k in ('area','image','mask','class_map','crs'))
                assert generated['x']==r['x'] and generated['y']==r['y']
            val_rows.append(dict(r,index_dir=str(folder)))
    print(name,'complete',flush=True)
records.sort(key=lambda r:(r['area'],r['image']))
assert sum(r['positive_count']>0 for r in records)==993
(OUT/'sampler-original.json').write_text(json.dumps(dict(source='Original A masks before coverage edits',tiles=records),indent=2)+'\n')
(OUT/'val-rows.json').write_text(json.dumps(val_rows,indent=2)+'\n')
print('sampler SHA256',hashlib.sha256((OUT/'sampler-original.json').read_bytes()).hexdigest(),flush=True)

(OUT/'sampler-treatment-drift.json').write_text(json.dumps(dict(original_positive_tiles=993,original_empty_tiles=752,changed=changed),indent=2)+'\n')
