import numpy as np
import pytest
import torch
from trailnet.fullarea import axis_jobs,assign_jobs,write_validity,predict_job
from trailnet.tta import TTA4,transform_input,invert_output

@pytest.mark.parametrize('phase',[0,192])
@pytest.mark.parametrize('length',[32,512,513,1000,7145])
def test_grid_ownership_exactly_once(length,phase):
    counts=np.zeros(length,int)
    for start,lo,hi in axis_jobs(length,phase=phase):
        assert 0<=lo<hi<=length and start<=lo and hi<=start+512
        counts[lo:hi]+=1
    assert np.all(counts==1)

def test_every_visible_device_receives_unique_jobs():
    jobs=list(range(10));groups=assign_jobs(jobs,2)
    assert groups==[[0,2,4,6,8],[1,3,5,7,9]]
    assert sorted(sum(groups,[]))==jobs
    with pytest.raises(ValueError):assign_jobs([1],2)

def test_metadata_masks_separate_and_cpu_prediction_smoke(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    transform=from_origin(500000,4500000,.6,.6)
    image=tmp_path/'optical.tif';dtm=tmp_path/'terrain.tif'
    raw=np.full((4,32,32),100,np.uint8);raw[:,2,2]=0
    z=np.ones((32,32),np.float32)*100;z[8,8]=-9999
    for path,data,nodata in [(image,raw,None),(dtm,z[None],-9999)]:
        with rasterio.open(path,'w',driver='GTiff',width=32,height=32,count=data.shape[0],dtype=data.dtype,crs='EPSG:32612',transform=transform,nodata=nodata) as out:
            out.write(data)
            if path==image:
                from rasterio.enums import ColorInterp
                out.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
                out.write_mask(np.full((32,32),255,np.uint8))  # explicit valid black pixel
    area=dict(name='fixture',imagery=str(image),dtm=str(dtm))
    counts=write_validity(area,tmp_path)
    assert counts['optical_invalid']==0 and counts['terrain_invalid']==1
    with rasterio.open(tmp_path/'fixture_optical_valid.tif') as src:assert src.read(1)[2,2]==1
    with rasterio.open(tmp_path/'fixture_terrain_valid.tif') as src:assert src.read(1)[8,8]==0
    class Model(torch.nn.Module):
        def forward(self,x):return x[:,:1]*2-1
    cfg=dict(tiling=dict(tile_size=512,overlap=128),terrain=dict(local_relief_radius_px=25,channel_gain=1.85),checkpoint_sha256='fixture')
    for arm,phase in [('A',0),('TTA4',0),('TTA4',192)]:
        result=predict_job(Model(),'cpu',cfg,area,arm,phase,tmp_path/f'{arm}{phase}',batch_size=1)
        with rasterio.open(result['probability']) as src:
            p=src.read(1);assert p.shape==(32,32) and np.isfinite(p).all()
            assert np.isclose(p[2,2],torch.sigmoid(torch.tensor(-1.)).item())
