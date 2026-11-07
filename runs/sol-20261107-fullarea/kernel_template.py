"""Self-contained, immutable full-area experiment; all allocated GPUs used."""
import sys,os,json,hashlib,base64,zipfile,io,subprocess,time,importlib.util
from pathlib import Path
MODE='__MODE__';PAYLOAD_SHA='__PAYLOAD_SHA__';PAYLOAD_B64='__PAYLOAD_B64__'
PROJECT=Path('/kaggle/temp/sol-fullarea-'+MODE)
OUT=Path('/kaggle/working/runs/sol-20261107-fullarea')/MODE
START=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def find(root,name):
    p=root/name
    return p if p.exists() else next(root.rglob(name))
def command(args,env=None):
    remaining=int(10800-(time.monotonic()-START))
    if remaining<=0:raise TimeoutError('Bounded full-area run exhausted')
    subprocess.run(args,env=env,check=True,timeout=remaining)
def main():
    OUT.mkdir(parents=True,exist_ok=True);PROJECT.mkdir(parents=True,exist_ok=True)
    data=base64.b64decode(PAYLOAD_B64);assert hashlib.sha256(data).hexdigest()==PAYLOAD_SHA
    with zipfile.ZipFile(io.BytesIO(data)) as z:z.extractall(PROJECT)
    import torch
    version,cuda=torch.__version__,torch.version.cuda
    missing=[p for m,p in {'rasterio':'rasterio','geopandas':'geopandas','yaml':'pyyaml','skimage':'scikit-image','scipy':'scipy','matplotlib':'matplotlib','segmentation_models_pytorch':'segmentation-models-pytorch'}.items() if importlib.util.find_spec(m) is None]
    if missing:
        constraint=OUT/'torch-constraints.txt';constraint.write_text(f'torch=={version}\n')
        command([sys.executable,'-m','pip','install','-q','-c',str(constraint),*missing])
    assert torch.__version__==version and torch.version.cuda==cuda and torch.cuda.is_available()
    print('ACTUAL_ALLOCATED_HARDWARE',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
    with (OUT/'runtime-packages.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True,timeout=120)
    root=Path('/kaggle/input');rasters=find(root,'trailnet-multiaoi-v4');code=find(root,'trailnet-t4e-code');ckpt=find(find(root,'trailnet-t4e2-a-ckpt'),'trailnet_best.pt')
    assert sha(ckpt)=='73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a'
    manifest=json.loads((PROJECT/'external-inputs-sha256.json').read_text())
    for path,expected in manifest.items():
        target=PROJECT/path
        if path.startswith('data/'):
            _,area,name=path.split('/');source=find(rasters,area+'_'+name)
        elif path.startswith('label_overrides/'):
            _,area,name=path.split('/');source=find(code,'label_overrides')/area/name
        else:source=target
        assert sha(source)==expected,(path,'frozen source hash mismatch')
        if source!=target:target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source)
    import yaml
    cfg=yaml.safe_load((PROJECT/'config_exp_t4e2_a_ignore5.yaml').read_text());cfg['_root']=str(PROJECT)
    cfg['checkpoint_sha256']=sha(ckpt);cfg['index_root']=str(PROJECT/'indexes');cfg['training']['checkpoint_path']=str(ckpt);cfg['model']['encoder_weights']=None
    cfg['inference']['threshold']=.9;cfg['postprocess']['threshold']=.9
    for area in cfg['areas']:
        name=area['name'];area['imagery']=str(PROJECT/'data'/name/'naip_rgbn.tif');area['dtm']=str(PROJECT/'data'/name/'3dep_dtm.tif');area['trails']=str(PROJECT/'label_overrides'/name/'trails.gpkg')
    (OUT/'frozen-config.json').write_text(json.dumps(cfg,indent=2));(OUT/'external-inputs-sha256.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'bundle.json').write_text(json.dumps(dict(mode=MODE,payload_sha256=PAYLOAD_SHA),indent=2))
    env=dict(os.environ,SOL_FULLAREA_CONFIG=str(OUT/'frozen-config.json'),SOL_FULLAREA_MODE=MODE,SOL_FULLAREA_OUTPUT=str(OUT))
    command([sys.executable,str(PROJECT/'runs/sol-20261107-fullarea/run_fullarea.py')],env)
    (OUT/'execution.json').write_text(json.dumps(dict(status='complete',mode=MODE,elapsed_seconds=time.monotonic()-START,payload_sha256=PAYLOAD_SHA,training=False),indent=2))
if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'execution.json').write_text(json.dumps(dict(status='failed',mode=MODE,error=repr(e),elapsed_seconds=time.monotonic()-START),indent=2));raise
