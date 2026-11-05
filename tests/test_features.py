"""The SAE dictionary, on whatever checkpoint exists. A young checkpoint is enough for these shapes."""
import os

import pytest

import neurospeak as ns

pytestmark = [pytest.mark.model, pytest.mark.skipif(not os.path.exists("runs/sae/Qwen__Qwen3.5-0.8B/L19.pt"), reason="no SAE trained yet")]


@pytest.fixture(scope="module")
def brain():
    return ns.load()


def test_features_read_text(brain):
    fs = brain.features[19]
    assert len(fs) == 16384
    hits = fs.active("So x = 5. Hmm, wait. So x = 5.", top=5)
    assert len(hits) == 5 and hits[0].value >= hits[-1].value > 0


def test_a_feature_finds_itself_and_can_be_erased(brain):
    f = brain.features[19][4021]
    assert brain.features[19].nearest(f.direction, top=1)[0].feature.index == 4021
    before = brain.params[brain.mlp_param(19)].detach().clone()
    with brain.edit() as e:
        e.erase(f.as_concept(), from_="L19.mlp")
        assert not brain.params[brain.mlp_param(19)].equal(before)
    assert brain.params[brain.mlp_param(19)].equal(before)


def test_why_keeps_the_model_unchanged_until_a_word_is_removed(brain):
    import torch
    from neurospeak.core import DEV
    from neurospeak.why import removed_logits
    ids = brain.tok("The capital of France is", return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():       # bfloat16 logits differ by rounding between grad and no-grad runs
        plain = brain.m(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    assert torch.allclose(removed_logits(brain, ids, 19, [[]]), plain)


def test_why_checks_each_word_by_removing_it(brain):
    report = brain.why("The capital of France is", top=4)
    assert report.answer == " Paris"
    assert report.words and all(w.measured is not None for w in report.words)
    assert report.gap_without is not None
    assert "removing all" in str(report)


def test_trace_links_a_word_to_what_feeds_it(brain):
    report = brain.trace("The capital of France is", top=2, parents_each=2)
    assert report.links and report.links[0].parents
    assert all(p.change != 0 for p in report.links[0].parents)
    assert "fed by" in str(report)


def test_diff_finds_no_change_for_no_edit_and_some_for_a_real_one(brain):
    texts = ["So x = 5. Hmm, wait. So x = 5. Hmm, wait. So x = 5."]
    report = brain.diff("runs/neurospeak_repetition_edit.pt", texts, top=3)
    assert report.tokens > 0 and len(report.falls) == 3
    assert report.change.abs().sum() > 0
    assert "fires less" in str(report)
