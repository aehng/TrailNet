"""Fetch exact A replay Park City windows from four September 2018 NAIP COGs."""
import base64,csv,hashlib,io,json,os,signal,time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio,requests
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

ROOT=Path(__file__).resolve().parent
OUT=Path('/kaggle/working/park-city-naip2018')
NAIP=Path('/kaggle/input/datasets/elikearl/trailnet-multiaoi-v4/park_city_naip_rgbn.tif')
ITEM_IDS=[
'ut_m_4011128_ne_12_060_20180927_20181209',
'ut_m_4011120_se_12_060_20180927_20181209',
'ut_m_4011128_nw_12_060_20180909_20181209',
'ut_m_4011120_sw_12_060_20180909_20181209']
SAMPLES='c2FtcGxlX2lkLGNlbnRlcl94LGNlbnRlcl95LGNyb3Bfd2lkdGhfcHgsY3JvcF9oZWlnaHRfcHgscmVzb2x1dGlvbl9tClcwMDEsNDQ5Njk0LjMwMDAwMDAwMDA1LDQ0OTY5MzUuNDk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDIsNDUyNDU5LjEwMDAwMDAwMDAzLDQ0OTY5MzUuNDk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDMsNDQ4NTQyLjMwMDAwMDAwMDA1LDQ0OTczOTYuMjk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDQsNDQ5MDAzLjEwMDAwMDAwMDAzLDQ0OTYwMTMuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDUsNDUyNjg5LjUwMDAwMDAwMDA2LDQ0OTcxNjUuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDYsNDUxMzA3LjEwMDAwMDAwMDAzLDQ0OTYwMTMuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDcsNDQ4NTQyLjMwMDAwMDAwMDA1LDQ0OTU1NTMuMSwyMTMsMjEzLDAuNgpXMDA4LDQ0OTY5NC4zMDAwMDAwMDAwNSw0NDk3MTY1Ljg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDA5LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk1MzIyLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEwLDQ1MDg0Ni4zMDAwMDAwMDAwNSw0NDk2OTM1LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDExLDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk0NjMxLjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEyLDQ0OTY5NC4zMDAwMDAwMDAwNSw0NDk1MzIyLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEzLDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk4Nzc4LjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE0LDQ1MTc2Ny45LDQ0OTc4NTcuMSwyMTMsMjEzLDAuNgpXMDE1LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk5MjM5LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE2LDQ1MDYxNS45LDQ0OTk3MDAuMjk5OTk5OTk5LDIxMywyMTMsMC42ClcwMTcsNDUwMTU1LjEwMDAwMDAwMDAzLDQ0OTkwMDkuMSwyMTMsMjEzLDAuNgpXMDE4LDQ1MDg0Ni4zMDAwMDAwMDAwNSw0NDk5NDY5Ljg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE5LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk3NjI2LjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIwLDQ1MTk5OC4zMDAwMDAwMDAwNSw0NDk5OTMwLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIxLDQ1MjQ1OS4xMDAwMDAwMDAwMyw0NDk4MDg3LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIyLDQ1MTk5OC4zMDAwMDAwMDAwNSw0NDk5NzAwLjI5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIzLDQ1MjY4OS41MDAwMDAwMDAwNiw0NDk0ODYxLjg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDI0LDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk5NzAwLjI5OTk5OTk5OSwyMTMsMjEzLDAuNgo='

def callback(status,**kw):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'callback.json').write_text(json.dumps({'status':status,**kw},indent=2),encoding='utf-8')
def timeout(_sig,_frame):
 callback('timed_out',timeout_seconds=1800)
 raise TimeoutError('30-minute runtime fallback')
def source_hash(): return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def item_metadata(session,item_id):
 url=f'https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/{item_id}'
 r=session.get(url,timeout=(15,45)); r.raise_for_status(); item=r.json()
 asset=item['assets']['image']; sign=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':asset['href']},timeout=(15,45)); sign.raise_for_status()
 return item,sign.json()['href']
def render_rgb(a):
 x=np.moveaxis(np.asarray(a[:3],dtype=np.float32),0,-1); v=x[np.isfinite(x)]
 if v.size:
  lo,hi=np.percentile(v,(1,99))
  if hi>lo: x=np.clip((x-lo)/(hi-lo),0,1)
 return x
def main():
 signal.signal(signal.SIGALRM,timeout); signal.alarm(1800)
 start=time.time(); OUT.mkdir(parents=True,exist_ok=True); (OUT/'chips').mkdir(exist_ok=True)
 samples=list(csv.DictReader(io.StringIO(base64.b64decode(SAMPLES).decode('utf-8-sig'))))
 session=requests.Session(); items=[]; signed=[]; metadata=[]
 for i,item_id in enumerate(ITEM_IDS,1):
  item,href=item_metadata(session,item_id); items.append(item); signed.append(href)
  metadata.append({'id':item['id'],'datetime':item.get('properties',{}).get('datetime'),'gsd':item.get('properties',{}).get('gsd'),'proj:epsg':item.get('properties',{}).get('proj:epsg'),'raster:bands':item['assets']['image'].get('raster:bands'),'eo:bands':item['assets']['image'].get('eo:bands'),'license':item.get('properties',{}).get('license','unknown'),'asset_type':item['assets']['image'].get('type'),'unsigned_href':item['assets']['image']['href'].split('?')[0]})
  callback('metadata_progress',items=i,item_id=item_id)
 opened=[]
 try:
  for href in signed:
   ds=rasterio.open(href)
   if ds.crs is None or ds.crs.to_epsg() not in (26912,32612): raise RuntimeError(f'Expected UTM zone 12 source; got {ds.crs}')
   opened.append(WarpedVRT(ds,crs='EPSG:32612',res=(0.6,0.6),nodata=0,resampling=Resampling.bilinear))
  current_panels=[]; records=[]
  with rasterio.open(NAIP) as current:
   for i,row in enumerate(samples,1):
    sid=row['sample_id']; cx,cy=float(row['center_x']),float(row['center_y']); wpx,hpx=int(row['crop_width_px']),int(row['crop_height_px']); res=float(row.get('resolution_m') or abs(current.transform.a)); w,h=wpx*res,hpx*res
    left,right=cx-w/2,cx+w/2; bottom,top=cy-h/2,cy+h/2; bbox=(left,bottom,right,top)
    win=Window(current.index(left,top)[1],current.index(left,top)[0],wpx,hpx)
    transform=rasterio.windows.transform(win,current.transform)
    cur=current.read([1,2,3,4],window=win,boundless=True,fill_value=0)
    cm=np.logical_and.reduce(current.read_masks([1,2,3,4],window=win,boundless=True)>0)
    candidate,ct=merge(opened,bounds=bbox,res=(res,res),indexes=[1,2,3,4],nodata=0,masked=True,method='first')
    cdata=np.asarray(candidate.filled(0)); valid=~np.logical_or.reduce(np.ma.getmaskarray(candidate),axis=0)
    if cdata.shape[1:]!=(hpx,wpx): raise RuntimeError(f'{sid}: candidate shape {cdata.shape} does not match {(hpx,wpx)}')
    zero=float(np.all(cdata[:3]==0,axis=0).mean()); vf=float(valid.mean())
    np.savez_compressed(OUT/'chips'/f'{sid}.npz',current=cur,current_valid=cm,candidate=cdata,candidate_valid=valid,bbox=np.asarray(bbox))
    current_panels.append((sid,render_rgb(cur),render_rgb(cdata),float(cm.mean()),vf,zero))
    records.append({'sample_id':sid,'center_epsg32612_m':[cx,cy],'bbox_epsg32612_m':list(bbox),'ground_window_m':[w,h],'current_mask_valid_fraction':round(float(cm.mean()),6),'candidate_mask_valid_fraction':round(vf,6),'candidate_zero_rgb_fraction':round(zero,6),'candidate_nonzero_rgb_fraction':round(1-zero,6),'candidate_zero_return_suspected':bool(zero>.99),'candidate_band_count':int(cdata.shape[0]),'candidate_transform':list(ct)[:6],'candidate_crs':'EPSG:32612'})
    callback('chip_progress',chips=i,total=len(samples),latest=sid)
  for start_i in range(0,len(current_panels),6):
   subset=current_panels[start_i:start_i+6]; fig,axes=plt.subplots(len(subset),2,figsize=(12,max(4,len(subset)*2.7)),squeeze=False)
   for j,(sid,cur,cand,cm,vf,zf) in enumerate(subset):
    axes[j,0].imshow(cur); axes[j,0].set_title(f'{sid} current NAIP 2021 | mask {cm:.1%}')
    axes[j,1].imshow(cand); axes[j,1].set_title(f'{sid} NAIP Sep 2018 | mask {vf:.1%}, zero {zf:.1%}')
    for ax in axes[j]: ax.set_xticks([]); ax.set_yticks([])
   fig.suptitle('Exact recovered-A Park City windows; images only, no labels or predictions')
   fig.tight_layout(); fig.savefig(OUT/f'paired-contact-sheet-{start_i//6+1:02d}.png',dpi=130); plt.close(fig)
  result={'status':'complete','sample_count':len(records),'sample_manifest_sha256':hashlib.sha256(base64.b64decode(SAMPLES)).hexdigest(),'source_code_sha256':source_hash(),'elapsed_seconds':round(time.time()-start,1),'candidate':'Planetary Computer NAIP Sept 2018 four-item mosaic','items':metadata,'current_source':'park_city_naip_rgbn.tif; 2021 September/November mosaic','current_colorinterp':'recorded separately in paired screen metadata','band_roles':'STAC item metadata is authoritative; Rasterio colorinterp may label the fourth band alpha. Inspect bands before trusting NIR interpretation.','ml_reuse_rights':'review dataset and Planetary Computer terms for intended use; no redistribution.','human_visibility_review':'pending; image-only sheets','snow_shadow_cloud_canopy_scored':False,'allzero_windows':sum(r['candidate_zero_return_suspected'] for r in records),'mean_zero_rgb_fraction':round(float(np.mean([r['candidate_zero_rgb_fraction'] for r in records])),6),'windows':records}
  (OUT/'source-items.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8'); (OUT/'paired-imagery-screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  callback('complete',sample_count=len(records),allzero_windows=result['allzero_windows'],elapsed_seconds=result['elapsed_seconds'])
  print(f"SCREEN_COMPLETE samples={len(records)} allzero={result['allzero_windows']} output={OUT}",flush=True)
 finally:
  for ds in opened: ds.close()
  signal.alarm(0)
if __name__=='__main__':
 try: main()
 except BaseException as e: callback('failed',error=repr(e)); raise
