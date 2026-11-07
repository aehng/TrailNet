"""Metadata-based source validity audit. Optical and terrain remain separate."""
import json,sys,warnings
from pathlib import Path
import numpy as np,rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from trailnet.data.tiles import _normalize_imagery
from trailnet.data.terrain import derive_terrain
OUT=Path(__file__).resolve().parent
SOURCE=Path('/workspace/trailnet-eval-vb-check/TrailNet/data')
rows=json.loads((OUT/'val-rows.json').read_text())
result={'method':'Declared nodata and GDAL masks/alpha only; no zero/black heuristic. Separate modalities. No truth/registration/weather inference.','areas':{}}
for folder in sorted(SOURCE.iterdir()):
    if not (folder/'naip_rgbn.tif').exists(): continue
    entry={}
    for kind,file in [('optical','naip_rgbn.tif'),('terrain','3dep_dtm.tif')]:
        with rasterio.open(folder/file) as src:
            invalid=np.zeros(src.count,np.int64); union=0; total=0
            for _,win in src.block_windows(1):
                masks=src.read_masks(window=win)
                invalid+=(masks==0).sum(axis=(1,2)); union+=int(np.any(masks==0,axis=0).sum()); total+=int(win.width*win.height)
            entry[kind]=dict(path=str(folder/file),nodata=src.nodata,colorinterp=[c.name for c in src.colorinterp],mask_flags=[[f.name for f in fs] for fs in src.mask_flag_enums],shape=[src.count,src.height,src.width],crs=str(src.crs),transform=list(src.transform),invalid_by_band=invalid.tolist(),any_band_invalid=union,total_pixels=total)
    # Trace two fixed VAL windows per available area, including edge if present.
    candidates=[r for r in rows if r['area']==folder.name]
    entry['val_traces']=[]
    if candidates:
        chosen=[candidates[0],candidates[-1]]
        for r in chosen:
            import re
            rr,cc=map(int,re.search(r'_r(\d+)_c(\d+)',r['image']).groups())
            win=Window(cc,rr,512,512)
            with rasterio.open(folder/'naip_rgbn.tif') as src,rasterio.open(folder/'3dep_dtm.tif') as dtm:
                raw=src.read([1,2,3,4],window=win,boundless=True,fill_value=0)
                norm=_normalize_imagery(raw)
                transform=src.window_transform(win)
                z=np.full((512,512),np.nan,np.float32)
                reproject(rasterio.band(dtm,1),z,src_transform=dtm.transform,src_crs=dtm.crs,src_nodata=dtm.nodata,dst_transform=transform,dst_crs=src.crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
                terr=np.stack(derive_terrain(z,src.transform.a,src.transform.e,25))*1.85
                with rasterio.open(Path(r['index_dir'])/r['image']) as tile: actual=tile.read()
                entry['val_traces'].append(dict(image=r['image'],optical_tensor_max_abs_difference=float(np.max(np.abs(norm-actual[:4]))),terrain_tensor_max_abs_difference=float(np.max(np.abs(terr-actual[4:]))),terrain_nonfinite_reprojected=int((~np.isfinite(z)).sum()),optical_declared_invalid_pixels=int(np.any(src.read_masks([1,2,3,4],window=win,boundless=True)==0,axis=0).sum())))
    result['areas'][folder.name]=entry
    print(folder.name,entry['optical']['invalid_by_band'],entry['terrain']['invalid_by_band'],flush=True)
(OUT/'validity-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
