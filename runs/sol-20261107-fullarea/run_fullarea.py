import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trailnet.fullarea import run
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['SOL_FULLAREA_CONFIG']).read_text())
    run(cfg,os.environ['SOL_FULLAREA_MODE'],os.environ['SOL_FULLAREA_OUTPUT'])
