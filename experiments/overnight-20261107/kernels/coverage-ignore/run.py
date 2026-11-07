"""One-seed diagnostic: ignore loss outside verified raster bounds. No heldout scoring."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path('/kaggle/working')
INPUT_ROOT = Path('/kaggle/input')
PROJECT = Path('/kaggle/temp/TrailNet-coverage-ignore')
CONFIG_NAME = 'config_exp_t4q_coverage_ignore.yaml'
RASTER_SLUG = 'trailnet-multiaoi-v4'
CODE_SLUG = 'trailnet-t4e-code'
AREAS = ['kelly_canyon', 'corner_canyon', 'park_city', 'teton_pass', 'snowbasin']
EXPECTED_INDEX = {
 'corner_canyon':'277eac2dad6bdc33386eba744838a82fccea0eea0cf9e90f69a83faee7ed2411',
 'kelly_canyon':'60bb9f2c139cfaa47c6de7fda1121d325bfb4807ff1c056ad2dbb2d295a3f63d',
 'park_city':'b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2',
 'snowbasin':'250e16679bacb0a0d20d1160ad85e1724718a20576e3b8a19535eeb126732500',
 'teton_pass':'f04eff6be5a833ad08d2a882c3e7a88eaf16b53cef4a948eea69ab735b50984f',
}
PREPARE_TIMEOUT_SECONDS = 30 * 60
TRAIN_TIMEOUT_SECONDS = 120 * 60
TOP_TIMEOUT_SECONDS = 135 * 60

def log(message: str) -> None:
    print(message, flush=True)
    with (WORK / 'coverage-ignore.log').open('a', encoding='utf-8') as f:
        f.write(message + '\n')

def callback(stage: str, status: str, **extra) -> None:
    folder = WORK / 'callbacks'
    folder.mkdir(parents=True, exist_ok=True)
    record = {'stage':stage,'status':status,
      'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
    path = folder / f'coverage-ignore-{stage}.callback.json'
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(record,indent=2),encoding='utf-8')
    tmp.replace(path)

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def find_mount(slug: str) -> Path:
    direct=INPUT_ROOT/slug
    if direct.is_dir(): return direct
    found=[p for p in INPUT_ROOT.rglob(slug) if p.is_dir()]
    if found: return found[0]
    raise FileNotFoundError(f'{slug} not found under {INPUT_ROOT}')

def find_file(root: Path,name: str) -> Path|None:
    direct=root/name
    if direct.is_file(): return direct
    return next(iter(root.rglob(name)),None)

def run_command(label: str,args: list[str],timeout: int) -> str:
    started=time.monotonic()
    callback(label,'running',arguments=args,timeout_seconds=timeout)
    proc=subprocess.Popen(args,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,errors='replace')
    try:
        output,_=proc.communicate(timeout=timeout)
        timed_out=False
    except subprocess.TimeoutExpired:
        proc.kill(); output,_=proc.communicate(); timed_out=True
    for line in output.splitlines(): log(line)
    status='timed_out' if timed_out else ('complete' if proc.returncode==0 else 'failed')
    callback(label,status,elapsed_seconds=round(time.monotonic()-started,3),
             timeout_seconds=timeout,exit_code=proc.returncode,
             timed_fallback_triggered=timed_out,output_tail=output[-6000:])
    if timed_out: raise TimeoutError(f'{label} exceeded {timeout}s')
    if proc.returncode: raise subprocess.CalledProcessError(proc.returncode,args,output=output)
    return output

def source_bounds_mask(transform,width,height,bounds):
    import numpy as np
    if transform.b != 0 or transform.d != 0:
        raise ValueError('rotated source/tile grids are unsupported')
    rr,cc=np.indices((height,width))
    x=transform.c+(cc+0.5)*transform.a
    y=transform.f+(rr+0.5)*transform.e
    left,bottom,right,top=bounds
    return (x<left)|(x>=right)|(y<bottom)|(y>=top)

def verify_outside_pixels(classes,target,outside):
    import numpy as np
    if classes.shape != target.shape or classes.shape != outside.shape:
        raise ValueError('class, target, and source-bounds mask shapes differ')
    bad_class=int(np.count_nonzero(classes[outside] != 4))
    bad_target=int(np.count_nonzero(target[outside] != 0))
    if bad_class or bad_target:
        raise RuntimeError(f'outside-footprint supervision remains: class={bad_class} target={bad_target}')
    return int(outside.sum())

def stage_inputs() -> dict:
    rasters,code_mount=find_mount(RASTER_SLUG),find_mount(CODE_SLUG)
    roots=[p for p in (code_mount,code_mount/'trailnet_code') if (p/'src').is_dir()]
    if not roots: raise FileNotFoundError('source tree is missing from Kaggle code dataset')
    src=roots[0]
    for name in ('src','scripts','requirements.txt','label_overrides'):
        item=src/name
        if item.is_dir(): shutil.copytree(item,PROJECT/name,dirs_exist_ok=True)
        elif item.is_file(): shutil.copy2(item,PROJECT/name)
    # Kaggle's code dataset predates this fix; inject only the committed tiler module.
    package_dir=Path(__file__).resolve().parent
    patch=package_dir/'tiles_coverage_patch.py'
    target=PROJECT/'src'/'trailnet'/'data'/'tiles.py'
    if not patch.is_file(): raise FileNotFoundError(f'coverage patch absent: {patch}')
    shutil.copy2(patch,target)
    patch_hash=sha256(target)
    expected_patch_hash='1256163013fb7001ecdb8d85d2cdd6cf6d10bb32ccd6b8eea53d3637c27f416b'
    if expected_patch_hash and patch_hash != expected_patch_hash:
        raise RuntimeError(f'injected tiles.py hash mismatch: {patch_hash}')
    cfg_source=package_dir/CONFIG_NAME
    shutil.copy2(cfg_source,PROJECT/CONFIG_NAME)
    config_hash=sha256(PROJECT/CONFIG_NAME)
    import yaml
    config=yaml.safe_load((PROJECT/CONFIG_NAME).read_text(encoding='utf-8'))
    if not config.get('tiling',{}).get('ignore_outside_imagery',False):
        raise RuntimeError('tiling.ignore_outside_imagery is not enabled')
    if config.get('seed') != 42 or config['training'].get('batch_size') != 12 or config['training'].get('epochs') != 16:
        raise RuntimeError('coverage run changed the A seed/batch/epoch recipe')
    for area in AREAS:
        folder=PROJECT/'data'/area; folder.mkdir(parents=True,exist_ok=True)
        for stem,suffix in (('naip_rgbn','.tif'),('3dep_dtm','.tif'),('trails','.gpkg')):
            raw=find_file(rasters,f'{area}_{stem}{suffix}')
            if raw is None: raise FileNotFoundError(f'missing raw input {area}_{stem}{suffix}')
            shutil.copy2(raw,folder/f'{stem}{suffix}')
        override=PROJECT/'label_overrides'/area/'trails.gpkg'
        if not override.is_file(): raise FileNotFoundError(f'missing A label override: {override}')
        shutil.copy2(override,folder/'trails.gpkg')
    return {'tiles_py_sha256':patch_hash,'config_sha256':config_hash,
            'source_code_dataset':'elikearl/trailnet-t4e-code','raw_dataset':'elikearl/trailnet-multiaoi-v4'}

def ensure_dependencies() -> dict:
    import torch
    version,cuda=torch.__version__,torch.version.cuda
    modules={'rasterio':'rasterio','geopandas':'geopandas','yaml':'pyyaml',
             'segmentation_models_pytorch':'segmentation-models-pytorch'}
    missing=[package for module,package in modules.items() if importlib.util.find_spec(module) is None]
    if missing:
        constraint=WORK/'torch-constraints.txt'
        constraint.write_text(f'torch=={version}\n',encoding='utf-8')
        run_command('dependencies',[sys.executable,'-m','pip','install','-q','-c',str(constraint),*missing],30*60)
    import torch as check
    if check.__version__ != version or check.version.cuda != cuda or not check.cuda.is_available():
        raise RuntimeError('Torch/CUDA changed or CUDA unavailable; refusing training')
    names=[check.cuda.get_device_name(i) for i in range(check.cuda.device_count())]
    if not names: raise RuntimeError('Kaggle GPU is unavailable')
    return {'torch':version,'cuda':cuda,'gpu_count':len(names),'gpus':names}

def prepare_and_verify() -> dict:
    import numpy as np
    import rasterio
    sys.path.insert(0,str(PROJECT/'src'))
    from trailnet.config import load_config,resolve_path
    from trailnet.data.dataset import read_indexes,spatial_split
    run_command('prepare-data',[sys.executable,'scripts/prepare_data.py','--config',CONFIG_NAME,'--all-areas'],PREPARE_TIMEOUT_SECONDS)
    cfg=load_config(PROJECT/CONFIG_NAME)
    idx_paths=[resolve_path(cfg,cfg['paths']['tile_dir'])/area/'index.csv' for area in AREAS]
    hashes={area:sha256(path) for area,path in zip(AREAS,idx_paths)}
    if hashes != EXPECTED_INDEX:
        raise RuntimeError(f'A index mismatch: {hashes}')
    rows=read_indexes(idx_paths)
    split=spatial_split(rows,block_size_m=cfg['training']['block_size_m'],
        val_fraction=cfg['training']['val_fraction'],test_fraction=cfg['training']['test_fraction'],
        seed=cfg.get('seed',42))
    counts={k:len(split[k]) for k in ('train','val','test')}
    if counts != {'train':1745,'val':172,'test':358}:
        raise RuntimeError(f'frozen A split counts mismatch: {counts}')
    edge={'train':0,'val':0,'test':0}; outside_total=0; outside_class4=0
    tile_root=resolve_path(cfg,cfg['paths']['tile_dir'])
    split_lookup={(row['area'],row['image']):name for name in ('train','val','test') for row in split[name]}
    source_bounds={}
    for area_cfg in cfg['areas']:
        name=area_cfg['name']
        with rasterio.open(resolve_path(cfg,area_cfg['imagery'])) as ds:
            source_bounds[name]=(ds.bounds.left,ds.bounds.bottom,ds.bounds.right,ds.bounds.top)
    for row in rows:
        area=row['area']; folder=tile_root/area
        with rasterio.open(folder/row['image']) as img:
            outside=source_bounds_mask(img.transform,img.width,img.height,source_bounds[area])
        if not outside.any(): continue
        split_name=split_lookup[(area,row['image'])]; edge[split_name]+=1
        with rasterio.open(folder/row['class_map']) as ds: classes=ds.read(1)
        with rasterio.open(folder/row['mask']) as ds: target=ds.read(1)
        outside_pixels=verify_outside_pixels(classes,target,outside)
        outside_total += outside_pixels; outside_class4 += int(np.count_nonzero(classes[outside]==4))
    return {'tile_index_hashes':hashes,'tile_counts_by_split':counts,
            'edge_tiles_by_split':edge,'outside_pixels_verified_as_ignore':outside_total,
            'outside_class4_pixels':outside_class4,'heldout_scored':False}

TRAINING_LAUNCHED = False

def timed_fallback():
    callback('final','timed_out',timeout_seconds=TOP_TIMEOUT_SECONDS,
      training_launched=TRAINING_LAUNCHED,heldout_scored=False,timed_fallback_triggered=True)
    os._exit(124)

def main():
    global TRAINING_LAUNCHED
    WORK.mkdir(parents=True,exist_ok=True); PROJECT.mkdir(parents=True,exist_ok=True)
    timer=threading.Timer(TOP_TIMEOUT_SECONDS,timed_fallback); timer.daemon=True; timer.start()
    started=time.monotonic(); callback('start','running',training_launched=False,timeout_seconds=TOP_TIMEOUT_SECONDS)
    try:
        runtime=ensure_dependencies()
        provenance=stage_inputs()
        preflight=prepare_and_verify()
        callback('preflight','complete',**preflight)
        log('COVERAGE_PREFLIGHT_OK '+json.dumps(preflight))
        # First train command occurs only after geometry-mask, config, and split checks pass.
        callback('train','running',timeout_seconds=TRAIN_TIMEOUT_SECONDS,seed=42,heldout_scored=False)
        TRAINING_LAUNCHED = True
        started_train=time.monotonic()
        output=run_command('train-command',[sys.executable,'scripts/train.py','--config',CONFIG_NAME,'--all-areas'],TRAIN_TIMEOUT_SECONDS)
        callback('train','complete',elapsed_seconds=round(time.monotonic()-started_train,2),heldout_scored=False)
        cfg_path=PROJECT/CONFIG_NAME
        import torch
        ckpt_path=PROJECT/'checkpoints_t4q_coverage_ignore'/'trailnet_best.pt'
        if not ckpt_path.is_file(): raise FileNotFoundError(f'training checkpoint missing: {ckpt_path}')
        ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        out=WORK/'coverage-ignore'; out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ckpt_path,out/'trailnet_best.pt'); shutil.copy2(cfg_path,out/CONFIG_NAME)
        result={'status':'complete','experiment':'t4q-coverage-ignore diagnostic','seed':42,
          'training_launched':True,'heldout_scored':False,'promotion_eligible':False,
          'promotion_blocked_without_independent_references':True,'provenance':provenance,
          'runtime':runtime,'preflight':preflight,'checkpoint_sha256':sha256(ckpt_path),
          'best_epoch':ckpt.get('epoch'),'coverage_aware_val_f1':ckpt.get('val_f1'),
          'coverage_aware_val_threshold':ckpt.get('best_threshold'),
          'checkpoint_val_counts':{'tp':ckpt.get('val_tp'),'fp':ckpt.get('val_fp'),'fn':ckpt.get('val_fn')},
          'elapsed_seconds':round(time.monotonic()-started,2),
          'train_log_tail':output[-12000:]}
        (out/'coverage-ignore-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        callback('final','complete',elapsed_seconds=result['elapsed_seconds'],training_launched=True,
          heldout_scored=False,timed_fallback_triggered=False,checkpoint_sha256=result['checkpoint_sha256'])
        log('COVERAGE_TRAIN_COMPLETE '+json.dumps({k:result[k] for k in ('best_epoch','coverage_aware_val_f1','coverage_aware_val_threshold','checkpoint_sha256')}))
    except Exception as exc:
        callback('final','failed',error=repr(exc),elapsed_seconds=round(time.monotonic()-started,2),
          training_launched=TRAINING_LAUNCHED,heldout_scored=False,timed_fallback_triggered=False)
        raise
    finally:
        timer.cancel()

if __name__=='__main__': main()
