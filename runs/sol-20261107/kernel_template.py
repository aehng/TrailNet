"""Self-contained frozen TTA comparison. One bounded GPU job; no training."""
import base64,hashlib,io,json,os,subprocess,sys,time,zipfile,importlib.util
from pathlib import Path
PAYLOAD_SHA='__PAYLOAD_SHA__'
PAYLOAD_B64='__PAYLOAD_B64__'
WORK=Path('/kaggle/working');PROJECT=Path('/kaggle/temp/sol-attempt');OUT=PROJECT/'runs/sol-20261107'
START=time.monotonic()
def run(args,env=None):
    remaining=int(45*60-(time.monotonic()-START))
    if remaining<=0: raise TimeoutError('45-minute bounded job exhausted')
    subprocess.run(args,env=env,check=True,timeout=remaining)
def find(root,name):
    direct=root/name
    return direct if direct.exists() else next(root.rglob(name))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    WORK.mkdir(exist_ok=True);PROJECT.mkdir(parents=True,exist_ok=True)
    payload=base64.b64decode(PAYLOAD_B64);assert hashlib.sha256(payload).hexdigest()==PAYLOAD_SHA
    with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(PROJECT)
    import torch
    version,cuda=torch.__version__,torch.version.cuda
    missing=[p for m,p in {'rasterio':'rasterio','geopandas':'geopandas','yaml':'pyyaml','segmentation_models_pytorch':'segmentation-models-pytorch'}.items() if importlib.util.find_spec(m) is None]
    if missing:
        constraint=WORK/'torch-constraints.txt';constraint.write_text(f'torch=={version}\n')
        run([sys.executable,'-m','pip','install','-q','-c',str(constraint),*missing])
    assert torch.__version__==version and torch.version.cuda==cuda and torch.cuda.is_available()
    root=Path('/kaggle/input');raster=find(root,'trailnet-multiaoi-v4');code=find(root,'trailnet-t4e-code');ckpt=find(find(root,'trailnet-t4e2-a-ckpt'),'trailnet_best.pt')
    manifest=json.loads((OUT/'external-inputs-sha256.json').read_text())
    for path,expected in manifest.items():
        target=PROJECT/path
        if path.startswith('data/'):
            _,area,name=path.split('/');src=find(raster,area+'_'+name)
        elif path.startswith('label_overrides/'):
            _,area,name=path.split('/');src=find(code,'label_overrides')/area/name
        else:src=target
        assert sha(src)==expected,(path,sha(src),expected)
        if src!=target:
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(src)
    env=dict(os.environ,SOL_SOURCE_ROOT=str(PROJECT/'data'),SOL_LABEL_ROOT=str(PROJECT/'label_overrides'),SOL_INDEX_ROOT=str(PROJECT/'indexes'),SOL_CHECKPOINT=str(ckpt))
    print('FROZEN_INPUTS_VERIFIED',PAYLOAD_SHA,flush=True)
    run([sys.executable,str(OUT/'prepare_review.py')],env)
    assert sha(OUT/'sampler-original.json')=='e7d4f41185ffc0d08af9512fce331c6de099d8e0adb9be8d2c0fa6c53bb240e0'
    run([sys.executable,str(OUT/'tta_val.py')],env)
    (WORK/'execution.json').write_text(json.dumps(dict(status='complete',elapsed_seconds=time.monotonic()-START,payload_sha256=PAYLOAD_SHA,training_launched=False,heldout_scored=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception as e:
        (WORK/'execution.json').write_text(json.dumps(dict(status='failed',error=repr(e),elapsed_seconds=time.monotonic()-START,training_launched=False,heldout_scored=False),indent=2)+'\n');raise
