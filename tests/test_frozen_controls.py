import hashlib
import json
import pytest
import torch
from trailnet.frozen_eval import frozen_counts
from trailnet.train import _frozen_sampler_counts, _make_pos_sampler


def test_frozen_legacy_targets_and_coverage_are_separate():
    # Outside coverage includes a true positive and a missed positive.
    p=torch.tensor([.95,.1,.95,.95,.0])
    y=torch.tensor([1.,1.,0.,1.,1.])
    w=torch.tensor([1.,1.,1.,0.,1.])
    outside=torch.tensor([True,True,False,False,False])
    assert frozen_counts(p,y,w,.9)==dict(tp=1,fp=1,fn=2)
    assert frozen_counts(p,y,w,.9,outside)==dict(tp=0,fp=1,fn=1)
    assert torch.equal(y,torch.tensor([1.,1.,0.,1.,1.]))
    with pytest.raises(ValueError): frozen_counts(p,y,w,.9,outside[:2])


def test_original_sampler_survives_treatment_and_order(tmp_path):
    rows=[dict(area='a',image='edge'),dict(area='a',image='inside')]
    path=tmp_path/'sampler.json'
    path.write_text(json.dumps(dict(tiles=[dict(**rows[0],positive_count=5),dict(**rows[1],positive_count=0)])))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    # Treatment may erase edge positives; eligibility still comes from original A.
    assert _frozen_sampler_counts(rows[::-1],path,sha)==[0,5]
    assert _make_pos_sampler(_frozen_sampler_counts(rows,path,sha),2)==[2,1]
    with pytest.raises(ValueError): _frozen_sampler_counts(rows,path,'bad')
    with pytest.raises(ValueError): _frozen_sampler_counts(rows[:1],path,sha)
    with pytest.raises(ValueError): _frozen_sampler_counts([rows[0],rows[0]],path,sha)
