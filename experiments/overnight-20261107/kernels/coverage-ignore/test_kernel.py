import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

PATH=Path('experiments/overnight-20261107/kernels/coverage-ignore/run.py')
SPEC=importlib.util.spec_from_file_location('coverage_ignore_kernel',PATH)
KERNEL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)

def test_coverage_mask_uses_source_geometry_not_pixel_values():
    transform=SimpleNamespace(a=1.0,b=0.0,c=0.0,d=0.0,e=-1.0,f=2.0)
    outside=KERNEL.source_bounds_mask(transform,3,2,(0.0,0.0,2.0,2.0))
    assert np.array_equal(outside,[[False,False,True],[False,False,True]])

def test_generated_outside_pixels_must_be_ignore_and_target_zero():
    classes=np.array([[4,1,3]])
    target=np.array([[0,1,1]])
    outside=np.array([[True,False,False]])
    assert KERNEL.verify_outside_pixels(classes,target,outside)==1

def test_preflight_rejects_remaining_outside_positive_supervision():
    classes=np.array([[1]])
    target=np.array([[1]])
    outside=np.array([[True]])
    try:
        KERNEL.verify_outside_pixels(classes,target,outside)
    except RuntimeError as error:
        assert 'outside-footprint supervision remains' in str(error)
    else:
        raise AssertionError('preflight must reject an unignored outside target')

def test_kernel_has_no_heldout_scoring_and_patches_exact_tiles_source():
    text=PATH.read_text(encoding='utf-8')
    assert '1256163013fb7001ecdb8d85d2cdd6cf6d10bb32ccd6b8eea53d3637c27f416b' in text
    assert "'heldout_scored':False" in text
    assert "tiling.ignore_outside_imagery" in text
