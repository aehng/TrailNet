import base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
files={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py'))}
files['config_exp_t4e2_a_ignore5.yaml']=(ROOT/'config_exp_t4e2_a_ignore5.yaml').read_bytes()
files['external-inputs-sha256.json']=(ROOT/'runs/sol-20261107/external-inputs-sha256.json').read_bytes()
for name in ('PLAN.md','run_fullarea.py'):files['runs/sol-20261107-fullarea/'+name]=(OUT/name).read_bytes()
indexes=Path('/workspace/TrailNet-handoff-20261106-1650/assets/recovered_A/data/tiles_t4e_a')
for p in sorted(indexes.glob('*/index.csv')):files['indexes/'+p.parent.name+'/index.csv']=p.read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(files.items()):
        entry=zipfile.ZipInfo(name,date_time=(2026,11,7,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,data)
payload=buf.getvalue();sha=hashlib.sha256(payload).hexdigest()
for mode in ('compare','phase'):
    folder=OUT/('kernel-'+mode);folder.mkdir(exist_ok=True)
    code=(OUT/'kernel_template.py').read_text().replace('__MODE__',mode).replace('__PAYLOAD_SHA__',sha).replace('__PAYLOAD_B64__',base64.b64encode(payload).decode())
    (folder/'run.py').write_text(code)
    slug='trailnet-sol-fullarea-'+mode+'-20261107'
    meta=dict(id='elikearl/'+slug,title=slug,code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['elikearl/trailnet-multiaoi-v4','elikearl/trailnet-t4e-code','elikearl/trailnet-t4e2-a-ckpt'],kernel_sources=[],competition_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(OUT/'bundle-manifest.json').write_text(json.dumps(dict(payload_sha256=sha,files={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}),indent=2)+'\n')
print('PACKAGED',sha,len(files),len(payload))
