import pytest
import torch
from trailnet.tta import TTA4,TTA8,transform_input,invert_output,ensemble_probabilities


def test_all_transforms_preserve_alignment_and_channels():
    image=torch.arange(7*5*9).reshape(1,7,5,9)
    for turns,flip in TTA8:
        assert torch.equal(invert_output(transform_input(image,turns,flip),turns,flip),image)
    assert set(TTA4).issubset(set(TTA8))
    assert len({transform_input(image,k,f).contiguous().numpy().tobytes() for k,f in TTA8})==8


def test_probability_ensemble_aligns_directional_predictions():
    class Pointwise(torch.nn.Module):
        def forward(self,x): return x[:,:1]*2-x[:,3:4]
    image=torch.randn(2,7,5,9)
    model=Pointwise()
    for transforms in (TTA4,TTA8):
        assert torch.allclose(ensemble_probabilities(model,image,transforms),model(image).sigmoid(),atol=1e-7)
    with pytest.raises(ValueError): ensemble_probabilities(model,image,())
