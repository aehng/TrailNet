"""Build self-contained Kaggle source bundle with immutable external input hashes."""
import base64,hashlib,io,json,zipfile,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--heldout',action='store_true');args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
source=Path('/workspace/trailnet-eval-vb-check/TrailNet/data');indexes=Path('/workspace/TrailNet-handoff-20261106-1650/assets/recovered_A/data/tiles_t4e_a');labels=Path('/workspace/kaggle-t4e/code/label_overrides')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest={}
for area in sorted(p.name for p in labels.iterdir() if (indexes/p.name/'index.csv').is_file()):
    for filename in ('naip_rgbn.tif','3dep_dtm.tif'):manifest[f'data/{area}/{filename}']=sha(source/area/filename)
    manifest[f'label_overrides/{area}/trails.gpkg']=sha(labels/area/'trails.gpkg')
    manifest[f'indexes/{area}/index.csv']=sha(indexes/area/'index.csv')
(OUT/'external-inputs-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={}
for p in sorted((ROOT/'src').rglob('*.py')): files[str(p.relative_to(ROOT))]=p.read_bytes()
files['config_exp_t4e2_a_ignore5.yaml']=(ROOT/'config_exp_t4e2_a_ignore5.yaml').read_bytes()
for filename in ('prepare_review.py','tta_val.py','PLAN.md','external-inputs-sha256.json','sampler-original.json'):
    files['runs/sol-20261107/'+filename]=(OUT/filename).read_bytes()
if args.heldout:
    for filename in ('tta_heldout.py','frozen-selection.json','val-output/tta-val-results.json'):
        files['runs/sol-20261107/'+filename]=(OUT/filename).read_bytes()
for area in sorted(p.name for p in labels.iterdir() if (indexes/p.name/'index.csv').is_file()):files[f'indexes/{area}/index.csv']=(indexes/area/'index.csv').read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(files.items()):
        info=zipfile.ZipInfo(name,date_time=(2026,11,7,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
payload=buf.getvalue();payload_sha=hashlib.sha256(payload).hexdigest()
runner=(OUT/'kernel_template.py').read_text().replace('__PAYLOAD_SHA__',payload_sha).replace('__PAYLOAD_B64__',base64.b64encode(payload).decode()).replace('__EVAL_SPLIT__','test' if args.heldout else 'val').replace('__EVAL_SCRIPT__','tta_heldout' if args.heldout else 'tta_val').replace('__HELDOUT_FLAG__',str(args.heldout))
kernel=OUT/('kernel-heldout' if args.heldout else 'kernel');kernel.mkdir(exist_ok=True);(kernel/'run.py').write_text(runner)
slug='trailnet-sol-tta-heldout-20261107' if args.heldout else 'trailnet-sol-tta-val-20261107'
meta=dict(id='elikearl/'+slug,title=slug,code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=['elikearl/trailnet-multiaoi-v4','elikearl/trailnet-t4e-code','elikearl/trailnet-t4e2-a-ckpt'],kernel_sources=[],competition_sources=[],model_sources=[],machine_shape='NvidiaTeslaT4')
(kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(OUT/('heldout-bundle-manifest.json' if args.heldout else 'kernel-bundle-manifest.json')).write_text(json.dumps(dict(payload_sha256=payload_sha,files={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}),indent=2)+'\n')
print('BUNDLE',payload_sha,len(payload),flush=True)
