"""Frozen full-area inference experiments, sharded across all visible GPUs."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling


def axis_jobs(length, size=512, overlap=128, phase=0):
    """Return input start and disjoint output ownership on the source grid."""
    stride=size-overlap
    starts=list(range(-phase,length,stride))
    result=[]
    for i,start in enumerate(starts):
        lo=max(0,start if i==0 else start+overlap//2)
        hi=min(length,start+size if i==len(starts)-1 else start+size-(overlap-overlap//2))
        if hi>lo:result.append((start,lo,hi))
    return result


def assign_jobs(jobs, device_count):
    if device_count<1 or len(jobs)<device_count:
        raise ValueError('Every visible device must receive actual inference work')
    return [jobs[d::device_count] for d in range(device_count)]


def write_validity(area, folder):
    """Export modality masks separately, using declared masks/nodata, never brightness."""
    with rasterio.open(area['imagery']) as im,rasterio.open(area['dtm']) as dtm:
        profile=im.profile.copy()
        for key in ('blockxsize','blockysize','tiled','interleave'):profile.pop(key,None)
        profile.update(count=1,dtype='uint8',nodata=None,compress='deflate')
        terrain_source=dtm.read_masks(1)>0
        terrain_source &= np.isfinite(dtm.read(1))
        terrain=np.zeros((im.height,im.width),np.uint8)
        reproject(terrain_source.astype('uint8'),terrain,src_transform=dtm.transform,src_crs=dtm.crs,
                  dst_transform=im.transform,dst_crs=im.crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest)
        counts={'optical_invalid':0,'terrain_invalid':int((terrain==0).sum()),'pixels':im.height*im.width,
                'optical_nodata':im.nodata,'terrain_nodata':dtm.nodata,
                'optical_colorinterp':[c.name for c in im.colorinterp],
                'optical_mask_flags':[[f.name for f in flags] for flags in im.mask_flag_enums],
                'note':'Band4 read numerically as frozen NIR; stale alpha metadata is not a new masking heuristic. GDAL declared masks retained as diagnostic. Terrain mask nearest-reprojected independently; predictions and frozen bounds-only scoring unchanged.'}
        for name in ('optical','terrain'):
            with rasterio.open(folder/f"{area['name']}_{name}_valid.tif",'w',**profile) as dst:
                for row in range(0,im.height,256):
                    win=Window(0,row,im.width,min(256,im.height-row))
                    valid=(np.all(im.read_masks([1,2,3,4],window=win)>0,axis=0).astype('uint8') if name=='optical' else terrain[row:row+int(win.height)])
                    if name=='optical':counts['optical_invalid']+=int((valid==0).sum())
                    dst.write(valid,1,window=win)
        return counts


def predict_job(model,device,config,area,arm,phase,folder,batch_size=4):
    import torch
    from trailnet.infer import _input_tile
    from trailnet.tta import inference_probabilities
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/f"{area['name']}_probability.tif"
    start=time.monotonic();tiles_done=0
    with rasterio.open(area['imagery']) as im,rasterio.open(area['dtm']) as dtm:
        size=int(config['tiling']['tile_size']);overlap=int(config['tiling']['overlap'])
        ys=axis_jobs(im.height,size,overlap,phase);xs=axis_jobs(im.width,size,overlap,phase)
        jobs=[(y,x) for y in ys for x in xs]
        profile=im.profile.copy()
        for key in ('blockxsize','blockysize','tiled','interleave'):profile.pop(key,None)
        profile.update(count=1,dtype='float32',nodata=None,compress='deflate')
        with rasterio.open(output,'w',**profile) as dst:
            dst.update_tags(threshold='0.90',ensemble=arm,grid_phase_px=str(phase),checkpoint_sha256=config['checkpoint_sha256'])
            for offset in range(0,len(jobs),batch_size):
                chunk=jobs[offset:offset+batch_size];images=[]
                for (row,_,_),(col,_,_) in chunk:
                    image,_=_input_tile(im,dtm,row,col,size,config['terrain']['local_relief_radius_px'])
                    image[4:7]*=float(config['terrain']['channel_gain']);images.append(image.astype('float32'))
                batch=torch.from_numpy(np.stack(images)).to(device)
                with torch.inference_mode():p=inference_probabilities(model,batch,'tta4' if arm=='TTA4' else 'none').cpu().numpy()[:,0]
                for ((row,y0,y1),(col,x0,x1)),prob in zip(chunk,p):
                    dst.write(prob[y0-row:y1-row,x0-col:x1-col],1,window=Window(x0,y0,x1-x0,y1-y0))
                tiles_done+=len(chunk)
                if offset%80==0:print('DEVICE_PROGRESS',device,area['name'],arm,phase,tiles_done,len(jobs),flush=True)
    return dict(area=area['name'],arm=arm,phase=phase,tiles=tiles_done,device=str(device),seconds=time.monotonic()-start,probability=str(output))


def gpu_worker(device_id,jobs,config,root):
    import torch
    from trailnet.models.unet import build_model
    torch.set_num_threads(2);torch.cuda.set_device(device_id);device=torch.device(f'cuda:{device_id}')
    print('DEVICE_START',device_id,torch.cuda.get_device_name(device_id),'jobs',len(jobs),flush=True)
    model_cfg=dict(config['model'],encoder_weights=None);model=build_model(model_cfg).to(device)
    model.load_state_dict(torch.load(config['training']['checkpoint_path'],map_location=device,weights_only=False)['model']);model.eval()
    log=[]
    for area,arm,phase in jobs:
        record=predict_job(model,device,config,area,arm,phase,Path(root)/f'{arm}-phase{phase}')
        log.append(record);print('DEVICE_JOB_COMPLETE',json.dumps(record),flush=True)
        Path(root,f'device-{device_id}.json').write_text(json.dumps(dict(device=device_id,name=torch.cuda.get_device_name(device_id),work=log),indent=2))


def overlays(area,prob_path,vector_path,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import box
    folder.mkdir(exist_ok=True)
    lines=gpd.read_file(vector_path);labels=gpd.read_file(area['trails'])
    manifest=[]
    with rasterio.open(area['imagery']) as im,rasterio.open(prob_path) as prob:
        lines=lines.to_crs(im.crs);labels=labels.to_crs(im.crs)
        for fy in (.2,.5,.8):
            for fx in (.2,.5,.8):
                row=max(0,min(im.height-512,int(im.height*fy)-256));col=max(0,min(im.width-512,int(im.width*fx)-256))
                win=Window(col,row,min(512,im.width),min(512,im.height));bounds=im.window_bounds(win);region=box(*bounds)
                raw=im.read([1,2,3],window=win);scale=np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype,np.integer) else 1.
                rgb=np.clip(np.moveaxis(raw,0,-1)/scale,0,1);p=prob.read(1,window=win)
                fig,axes=plt.subplots(1,2,figsize=(10,5));extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
                axes[0].imshow(rgb,extent=extent);axes[1].imshow(p,vmin=0,vmax=1,cmap='magma',extent=extent)
                for ax in axes:
                    selected=lines[lines.intersects(region)]
                    if not selected.empty:selected.plot(ax=ax,color='cyan',linewidth=.7)
                    ref=labels[labels.intersects(region)]
                    if not ref.empty:ref.plot(ax=ax,color='gold',linewidth=.5,alpha=.7)
                    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_axis_off()
                fig.suptitle(f"{area['name']} fixed grid {fx},{fy}; cyan=unfiltered predictions; gold=unverified labels")
                path=folder/f"{area['name']}_{fx}_{fy}.png";fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig)
                manifest.append(dict(area=area['name'],fx=fx,fy=fy,row=row,col=col,path=str(path)))
    return manifest


def stability(area,left,right,validity,config):
    """VAL-only consistency on independent modality-valid intersection; no new GT."""
    from trailnet.vector_eval import assign_block_split
    totals=dict(valid_pixels=0,abs_diff_sum=0.,intersection=0,union=0)
    with rasterio.open(left) as a,rasterio.open(right) as b,rasterio.open(validity/f"{area['name']}_optical_valid.tif") as optical,rasterio.open(validity/f"{area['name']}_terrain_valid.tif") as terrain:
        for row in range(0,a.height,256):
            win=Window(0,row,a.width,min(256,a.height-row));pa=a.read(1,window=win);pb=b.read(1,window=win)
            xx=a.transform.c+(np.arange(a.width)+.5)*a.transform.a;yy=a.transform.f+(row+np.arange(int(win.height))+.5)*a.transform.e
            bx=np.floor(xx/2500).astype(int);by=np.floor(yy/2500).astype(int);val=np.zeros(pa.shape,bool)
            for y in np.unique(by):
                for x in np.unique(bx):
                    if assign_block_split(area['name'],int(x),int(y),seed=42,val_fraction=.15,test_fraction=.15)=='val':val|=(by[:,None]==y)&(bx[None,:]==x)
            valid=val&(optical.read(1,window=win)>0)&(terrain.read(1,window=win)>0)
            totals['valid_pixels']+=int(valid.sum());totals['abs_diff_sum']+=float(np.abs(pa-pb)[valid].sum(dtype=np.float64))
            totals['intersection']+=int(((pa>=.9)&(pb>=.9)&valid).sum());totals['union']+=int((((pa>=.9)|(pb>=.9))&valid).sum())
    totals['mae']=totals['abs_diff_sum']/totals['valid_pixels'] if totals['valid_pixels'] else None
    totals['jaccard']=totals['intersection']/totals['union'] if totals['union'] else None
    return totals


def run(config,mode,root):
    import torch,multiprocessing as mp
    from trailnet.postprocess import vectorize_probability
    from trailnet.vector_eval import vector_metrics_for_area,length_weighted_aggregate
    from trailnet.data.dataset import read_indexes
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    n=torch.cuda.device_count();assert n>0
    hardware=dict(device_count=n,devices=[torch.cuda.get_device_name(d) for d in range(n)],torch=torch.__version__,cuda=torch.version.cuda)
    print('HARDWARE',json.dumps(hardware),flush=True);(root/'hardware.json').write_text(json.dumps(hardware,indent=2))
    arms=[('A',0),('TTA4',0)] if mode=='compare' else [('TTA4',0),('TTA4',192)]
    jobs=[(a,arm,phase) for arm,phase in arms for a in config['areas']];groups=assign_jobs(jobs,n)
    ctx=mp.get_context('spawn');processes=[]
    for d,group in enumerate(groups):
        p=ctx.Process(target=gpu_worker,args=(d,group,config,str(root)));p.start();processes.append(p)
    validity=root/'validity';validity.mkdir(exist_ok=True)
    masks={a['name']:write_validity(a,validity) for a in config['areas']}
    (root/'validity-summary.json').write_text(json.dumps(masks,indent=2))
    for p in processes:p.join();assert p.exitcode==0,f'GPU worker failed: {p.exitcode}'
    metrics={};panels=[]
    for arm,phase in arms:
        name=f'{arm}-phase{phase}';folder=root/name;cfg=dict(config);cfg['paths']=dict(config['paths'],prediction_dir=str(folder),vector_dir=str(folder))
        metrics[name]={}
        for area in config['areas']:
            area_name=area['name'];prob=folder/f'{area_name}_probability.tif';vectors=folder/f'{area_name}_trails.geojson'
            vectorize_probability(prob,vectors,.9,1.,area['trails'],3.,area_name,novelty_filter=False,novelty_output_path=folder/f'{area_name}_novelty.geojson')
            rows=read_indexes([Path(config['index_root'])/area_name/'index.csv'])
            metrics[name][area_name]=vector_metrics_for_area(cfg,area,predicted_path=vectors,index_rows=rows,tolerances_m=[3.,5.,8.,10.],scopes=('val_blocks','test_blocks'))
            panels+=overlays(area,prob,vectors,folder/'overlays')
            (root/'vector-metrics.json').write_text(json.dumps(metrics,indent=2))
        metrics[name]['aggregate_val_5m']=length_weighted_aggregate({a['name']:metrics[name][a['name']]['scopes']['val_blocks'] for a in config['areas']},report_tolerance_m=5.)
    (root/'vector-metrics.json').write_text(json.dumps(metrics,indent=2))
    consistency={}
    if mode=='phase':
        consistency={a['name']:stability(a,root/'TTA4-phase0'/f"{a['name']}_probability.tif",root/'TTA4-phase192'/f"{a['name']}_probability.tif",validity,config) for a in config['areas']}
    support={}
    for name in metrics:
        m=metrics[name]['aggregate_val_5m'];p=m['precision'];r=m['recall']
        m['f1']=2*p*r/(p+r) if p is not None and r is not None and p+r else None
    if mode=='compare':
        a=metrics['A-phase0']['aggregate_val_5m'];b=metrics['TTA4-phase0']['aggregate_val_5m']
        support=dict(criterion='VAL vector F1 +.02 and recall >= A-.02',passed=(b['f1']>=a['f1']+.02 and b['recall']>=a['recall']-.02) if a['f1'] is not None and b['f1'] is not None else None)
    else:
        pooled={k:sum(v[k] for v in consistency.values()) for k in ('valid_pixels','abs_diff_sum','intersection','union')}
        pooled['mae']=pooled['abs_diff_sum']/pooled['valid_pixels'] if pooled['valid_pixels'] else None
        pooled['jaccard']=pooled['intersection']/pooled['union'] if pooled['union'] else None
        consistency['pooled_val']=pooled
        a=metrics['TTA4-phase0']['aggregate_val_5m'];b=metrics['TTA4-phase192']['aggregate_val_5m']
        support=dict(criterion='VAL vector P/R within .02; declared-valid VAL binary Jaccard >=.60',passed=(abs(a['precision']-b['precision'])<=.02 and abs(a['recall']-b['recall'])<=.02 and pooled['jaccard']>=.60) if a['recall'] is not None and b['recall'] is not None and pooled['jaccard'] is not None else None)
    records=[json.loads((root/f'device-{d}.json').read_text()) for d in range(n)];assert all(r['work'] for r in records)
    result=dict(status='complete',mode=mode,hardware=hardware,per_device_work=records,metrics=metrics,consistency=consistency,predeclared_support=support,
                elapsed_seconds=time.monotonic()-start,estimated_allocated_gpu_hours=n*(time.monotonic()-start)/3600,user_compute_cost_usd=0.,
                cost_note='Free Kaggle compute; allocated-device wall estimate, not authoritative quota billing.',
                heldout_note='All test-block numbers descriptive only: heldout already used; overlapping split tile footprints limit independence. Full AOI includes train geography. Unmatched predictions are not proven false trails.',
                decision_scope='VAL blocks only. Frozen threshold0.90; no training, threshold tuning or promotion from heldout.',overlays=panels)
    (root/'results.json').write_text(json.dumps(result,indent=2));print('FULLAREA_COMPLETE',mode,flush=True)
