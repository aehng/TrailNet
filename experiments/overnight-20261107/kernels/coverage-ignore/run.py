"""One-seed diagnostic: ignore loss outside verified raster bounds. No heldout scoring."""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import zlib
import time
from pathlib import Path

PATCH_SOURCE_ZLIB_B64 = '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'
CONFIG_YAML_B64 = 'c2VlZDogNDIKcGF0aHM6CiAgZGF0YV9kaXI6IGRhdGEKICB0aWxlX2RpcjogZGF0YS90aWxlc190NHFfY292ZXJhZ2VfaWdub3JlCiAgb3V0cHV0X2Rpcjogb3V0cHV0c190NHFfY292ZXJhZ2VfaWdub3JlCiAgY2hlY2twb2ludF9kaXI6IGNoZWNrcG9pbnRzX3Q0cV9jb3ZlcmFnZV9pZ25vcmUKICBwcmVkaWN0aW9uX2Rpcjogb3V0cHV0c190NHFfY292ZXJhZ2VfaWdub3JlL3ByZWRpY3Rpb25zCiAgdmVjdG9yX2Rpcjogb3V0cHV0c190NHFfY292ZXJhZ2VfaWdub3JlL3ZlY3RvcnMKYXJlYXM6Ci0gbmFtZToga2VsbHlfY2FueW9uCiAgZGVzY3JpcHRpb246IEV4aXN0aW5nIEtlbGx5IENhbnlvbiBOQUlQIGFuZCAzREVQIHdpbmRvdwogIGJib3g6CiAgLSAtMTExLjYyOTQKICAtIDQzLjYxNwogIC0gLTExMS41NzQKICAtIDQzLjY1NzgKICBpbWFnZXJ5OiBkYXRhL2tlbGx5X2Nhbnlvbi9uYWlwX3JnYm4udGlmCiAgZHRtOiBkYXRhL2tlbGx5X2Nhbnlvbi8zZGVwX2R0bS50aWYKICB0cmFpbHM6IGRhdGEva2VsbHlfY2FueW9uL3RyYWlscy5ncGtnCiAgbGFiZWxfc291cmNlczoKICAgIG9zbTogdHJ1ZQogICAgdWdyYzogZmFsc2UKICAgIHVzZnM6IHRydWUKLSBuYW1lOiBjb3JuZXJfY2FueW9uCiAgZGVzY3JpcHRpb246IERyYXBlciBDb3JuZXIgQ2FueW9uIHBhdGggYW5kIHNpbmdsZXRyYWNrIGNsdXN0ZXIKICBiYm94OgogIC0gLTExMS44NwogIC0gNDAuNDkKICAtIC0xMTEuODIKICAtIDQwLjU0CiAgaW1hZ2VyeTogZGF0YS9jb3JuZXJfY2FueW9uL25haXBfcmdibi50aWYKICBkdG06IGRhdGEvY29ybmVyX2Nhbnlvbi8zZGVwX2R0bS50aWYKICB0cmFpbHM6IGRhdGEvY29ybmVyX2Nhbnlvbi90cmFpbHMuZ3BrZwogIGxhYmVsX3NvdXJjZXM6CiAgICBvc206IHRydWUKICAgIHVncmM6IHRydWUKICAgIHVzZnM6IGZhbHNlCi0gbmFtZTogcGFya19jaXR5CiAgZGVzY3JpcHRpb246IFdhc2F0Y2ggQ3Jlc3QgYW5kIFBDTVIgdHJhaWwgY2x1c3RlcgogIGJib3g6CiAgLSAtMTExLjYxCiAgLSA0MC42CiAgLSAtMTExLjU2CiAgLSA0MC42NQogIGltYWdlcnk6IGRhdGEvcGFya19jaXR5L25haXBfcmdibi50aWYKICBkdG06IGRhdGEvcGFya19jaXR5LzNkZXBfZHRtLnRpZgogIHRyYWlsczogZGF0YS9wYXJrX2NpdHkvdHJhaWxzLmdwa2cKICBsYWJlbF9zb3VyY2VzOgogICAgb3NtOiB0cnVlCiAgICB1Z3JjOiB0cnVlCiAgICB1c2ZzOiBmYWxzZQotIG5hbWU6IHRldG9uX3Bhc3MKICBkZXNjcmlwdGlvbjogVGV0b24gUGFzcyB0cmFpbCBjbHVzdGVyCiAgYmJveDoKICAtIC0xMTEuMDMKICAtIDQzLjQ4CiAgLSAtMTEwLjk4CiAgLSA0My41MwogIGltYWdlcnk6IGRhdGEvdGV0b25fcGFzcy9uYWlwX3JnYm4udGlmCiAgZHRtOiBkYXRhL3RldG9uX3Bhc3MvM2RlcF9kdG0udGlmCiAgdHJhaWxzOiBkYXRhL3RldG9uX3Bhc3MvdHJhaWxzLmdwa2cKICBsYWJlbF9zb3VyY2VzOgogICAgb3NtOiB0cnVlCiAgICB1Z3JjOiBmYWxzZQogICAgdXNmczogdHJ1ZQotIG5hbWU6IHNub3diYXNpbgogIGRlc2NyaXB0aW9uOiBTbm93YmFzaW4gdHJhaWwgY2x1c3RlcgogIGJib3g6CiAgLSAtMTExLjg5CiAgLSA0MS4xOAogIC0gLTExMS44NAogIC0gNDEuMjMKICBpbWFnZXJ5OiBkYXRhL3Nub3diYXNpbi9uYWlwX3JnYm4udGlmCiAgZHRtOiBkYXRhL3Nub3diYXNpbi8zZGVwX2R0bS50aWYKICB0cmFpbHM6IGRhdGEvc25vd2Jhc2luL3RyYWlscy5ncGtnCiAgbGFiZWxfc291cmNlczoKICAgIG9zbTogdHJ1ZQogICAgdWdyYzogdHJ1ZQogICAgdXNmczogZmFsc2UKdGlsaW5nOgogIHRpbGVfc2l6ZTogNTEyCiAgb3ZlcmxhcDogMTI4CiAgcmVzb2x1dGlvbl9tOiAwLjYKICBpZ25vcmVfYnVmZmVyX206IDUuMAogIHNvZnRfbGFiZWxfYmFuZF9tOiAwLjAKICBzaWRld2Fsa191bmNlcnRhaW5fYnVmZmVyX206IDIuMAogIHRyYWlsX2J1ZmZlcl9tOgogICAgcGF0aDogMS4wCiAgICBmb290d2F5OiAxLjAKICAgIGN5Y2xld2F5OiAxLjAKICAgIHRyYWNrOiAxLjUKICAgIGRlZmF1bHQ6IDEuMAogIGlnbm9yZV9vdXRzaWRlX2ltYWdlcnk6IHRydWUKdGVycmFpbjoKICBsb2NhbF9yZWxpZWZfcmFkaXVzX3B4OiAyNQogIGNoYW5uZWxfZ2FpbjogMS44NQptb2RlbDoKICBhcmNoaXRlY3R1cmU6IGR1YWxfdGVycmFpbgogIGVuY29kZXJfbmFtZTogcmVzbmV0MzQKICB0ZXJyYWluX2VuY29kZXJfbmFtZTogcmVzbmV0MTgKICBlbmNvZGVyX3dlaWdodHM6IGltYWdlbmV0CiAgaW5fY2hhbm5lbHM6IDcKICBjbGFzc2VzOiAxCiAgZGlsYXRlZF9jZW50ZXI6IGZhbHNlCmxvc3M6CiAgbmFtZTogZm9jYWxfYmNlX3R2ZXJza3kKICBwb3NfYm9vc3Q6IDIuMAogIHBpeGVsX3dlaWdodHM6CiAgICBiYWNrZ3JvdW5kOiAxLjAKICAgIHBhdGg6IDguMAogICAgdHJhY2s6IDMuMAogICAgdW5jZXJ0YWluOiA0LjAKdHJhaW5pbmc6CiAgYmF0Y2hfc2l6ZTogMTIKICBhYm9ydF9pZl9iYXRjaF9ub3Q6IDEyCiAgZXBvY2hzOiAxNgogIGxlYXJuaW5nX3JhdGU6IDAuMDAwMQogIHdlaWdodF9kZWNheTogMC4wMDAxCiAgdmFsX2ZyYWN0aW9uOiAwLjE1CiAgdGVzdF9mcmFjdGlvbjogMC4xNQogIGJsb2NrX3NpemVfbTogMjUwMAogIG51bV93b3JrZXJzOiAyCiAgbWl4ZWRfcHJlY2lzaW9uOiB0cnVlCiAgd2FybXVwX2Vwb2NoczogMwogIHBvc19vdmVyc2FtcGxlOiAyLjAKICBlYXJseV9zdG9wcGluZ19wYXRpZW5jZTogNQogIGVhcmx5X3N0b3BwaW5nX21pbl9kZWx0YTogMC4wCiAgd3JpdGViYWNrX3RocmVzaG9sZDogdHJ1ZQogIHZhbF90aHJlc2hvbGRzOgogIC0gMC4yNQogIC0gMC4zCiAgLSAwLjM1CiAgLSAwLjQKICAtIDAuNDUKICAtIDAuNQogIC0gMC41NQogIC0gMC42CiAgLSAwLjY1CiAgLSAwLjcKICAtIDAuNzUKICAtIDAuOAogIC0gMC44NQogIC0gMC45CiAgY2hlY2twb2ludF9wYXRoOiBjaGVja3BvaW50c190NHFfY292ZXJhZ2VfaWdub3JlL3RyYWlsbmV0X2Jlc3QucHQKaW5mZXJlbmNlOgogIHRocmVzaG9sZDogMC42CiAgYmF0Y2hfc2l6ZTogOApwb3N0cHJvY2VzczoKICB0aHJlc2hvbGQ6IDAuNgogIHNpbXBsaWZ5X206IDEuMAogIGRlZHVwX3RvbGVyYW5jZV9tOiAzLjAKZG93bmxvYWRzOgogIG1heF9yYXN0ZXJfa20yOiAyNS4wCiAgbWF4X3Jhc3Rlcl9waXhlbHM6IDcwMDAwMDAwCiAgbWF4X2Rvd25sb2FkX2diOiAyLjAKICBtYXhfZXh0cmFjdGVkX2diOiA4LjAKICBtYXhfM2RlcF9wcm9kdWN0czogMjAwCiAgbWF4X29zbV9rbTI6IDUwMDAuMAogIG5haXBfc3RhY191cmw6IGh0dHBzOi8vcGxhbmV0YXJ5Y29tcHV0ZXIubWljcm9zb2Z0LmNvbS9hcGkvc3RhYy92MQogIG5haXBfY29sbGVjdGlvbjogbmFpcAogIG5haXBfYXNzZXQ6IGltYWdlCiAgcGxhbmV0YXJ5X2NvbXB1dGVyX3NpZ25fdXJsOiBodHRwczovL3BsYW5ldGFyeWNvbXB1dGVyLm1pY3Jvc29mdC5jb20vYXBpL3Nhcy92MS9zaWduCiAgdG5tX3Byb2R1Y3RzX3VybDogaHR0cHM6Ly90bm1hY2Nlc3MubmF0aW9uYWxtYXAuZ292L2FwaS92MS9wcm9kdWN0cwogIHRubV8xbV9kYXRhc2V0OiBEaWdpdGFsIEVsZXZhdGlvbiBNb2RlbCAoREVNKSAxIG1ldGVyCiAgZHRtX3Jlc29sdXRpb25fbTogMS4wCiAgb3ZlcnBhc3NfdXJsOiBodHRwczovL292ZXJwYXNzLWFwaS5kZS9hcGkvaW50ZXJwcmV0ZXIKICBvdmVycGFzc190b2tlbl9lbnY6IFRSQUlMTkVUX09WRVJQQVNTX1RPS0VOCiAgY2FjaGVfZGlyOiBkYXRhL19jYWNoZQogIHVncmNfdHJhaWxzX3VybDogaHR0cHM6Ly9zZXJ2aWNlczEuYXJjZ2lzLmNvbS85OWxpZFBoV0N6ZnRJZTlLL0FyY0dJUy9yZXN0L3NlcnZpY2VzL1RyYWlsc0FuZFBhdGh3YXlzL0ZlYXR1cmVTZXJ2ZXIvMC9xdWVyeQogIHVzZnNfdHJhaWxzX3VybDogaHR0cHM6Ly9hcHBzLmZzLnVzZGEuZ292L2FyY3gvcmVzdC9zZXJ2aWNlcy9FRFcvRURXX1RyYWlsTkZTUHVibGlzaF8wMS9NYXBTZXJ2ZXIvMC9xdWVyeQpsYWJlbHM6CiAgZHJvcF91bm5hbWVkX29zbV9mb290d2F5OiBmYWxzZQogIGRyb3Bfb3NtX2Zvb3R3YXk6IGZhbHNlCiAgcmVidWlsZF9mcm9tX3NvdXJjZXM6IGZhbHNlCiAgc25hcF90b19yZWxpZWY6CiAgICBlbmFibGVkOiBmYWxzZQpldmFsdWF0aW9uOgogIHZlY3Rvcl9tYXRjaF90b2xlcmFuY2VzX206CiAgLSAzLjAKICAtIDUuMAogIC0gOC4wCiAgLSAxMC4wCiAgdmVjdG9yX3JlcG9ydF90b2xlcmFuY2VfbTogNS4wCiAgY2xpcF92ZWN0b3JfbWV0cmljc190b19wcmVkaWN0aW9uOiB0cnVlCg=='
EXPECTED_TILES_SHA256 = '1256163013fb7001ecdb8d85d2cdd6cf6d10bb32ccd6b8eea53d3637c27f416b'

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
    patch_bytes=zlib.decompress(base64.b64decode(PATCH_SOURCE_ZLIB_B64))
    target=PROJECT/'src'/'trailnet'/'data'/'tiles.py'
    target.write_bytes(patch_bytes)
    patch_hash=sha256(target)
    if patch_hash != EXPECTED_TILES_SHA256:
        raise RuntimeError(f'injected tiles.py hash mismatch: {patch_hash}')
    config_bytes=base64.b64decode(CONFIG_YAML_B64)
    config_path=PROJECT/CONFIG_NAME
    config_path.write_bytes(config_bytes)
    config_hash=sha256(config_path)
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
