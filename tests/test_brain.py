"""Runs the real model: what a colleague importing the library can observe."""
import pytest
import torch

import neurospeak as ns

pytestmark = pytest.mark.model
STUCK = "So x = 5. Hmm, wait. So x = 5. Hmm, wait. So x = 5. Hmm, wait. So x = 5."
FINE = "So x = 5. Check: 3*5+7 = 22, which is correct. The answer is 5."
PAIRS = [ns.pair(is_=STUCK, not_=FINE, ask="Solve 3x + 7 = 22."),
         ns.pair(is_="4*180. So I need 4*180. So I need 4*180. So I need 4*180.",
                 not_="4*180 = 720 degrees, which also matches 4 triangles of 180.",
                 ask="What is the sum of the interior angles of a hexagon?")]


@pytest.fixture(scope="module")
def brain():
    return ns.load(ask_suffix=ns.MATH_SUFFIX)


@pytest.fixture(scope="module")
def rep(brain):
    return brain.concept("rep", PAIRS, at=[15, 19])


def weights(brain, layer=19):
    return brain.params[brain.mlp_param(layer)].detach().clone()


def test_edit_block_restores_weights(brain, rep):
    before = weights(brain)
    with brain.edit(strict=False) as e:
        e.erase(rep, from_=brain.L[15, 19].mlp, writers=64, strength=6)
        assert not torch.equal(weights(brain), before)
    assert torch.equal(weights(brain), before)
    assert brain.history == []


def test_keep_persists_until_undo(brain, rep):
    before = weights(brain)
    with brain.edit(keep=True) as e:
        e.erase(rep, from_="L19.mlp", writers=8)
    assert not torch.equal(weights(brain), before)
    assert brain.undo().startswith("erase rep")
    assert torch.equal(weights(brain), before)


def test_broken_promise_raises_and_still_restores(brain, rep):
    before = weights(brain)
    with pytest.raises(ns.ExpectationFailed) as err:
        with brain.edit() as e:
            e.erase(rep, from_="L19.mlp", writers=64, strength=6)
            e.expect(rep.is_).rises()
    assert len(err.value.broken) == 1
    assert torch.equal(weights(brain), before)


def test_promise_works_with_plain_assert(brain, rep):
    with brain.edit() as e:
        e.erase(rep, from_=brain.L[15, 19].mlp, writers=64, strength=6)
        assert e.expect(rep.is_).falls()


def test_writers_are_neurons_you_can_loop_over(brain, rep):
    top = rep.writers(brain.L[19].mlp, top=3)
    assert len(top) == 3 and all(isinstance(w.neuron, ns.Neuron) and w.neuron.layer == 19 for w in top)
    assert "who writes rep in L19.mlp" in str(top)


def test_where_readout_is_indexable(brain, rep):
    w = rep.where()
    assert len(w.by_layer) == brain.n_layers and w[w.best] == max(w.by_layer)


def test_concept_algebra(brain, rep):
    other = brain.concept("other", [ns.pair(is_="Yes. It is true.", not_="No. It is false.")])
    diff = rep - other
    assert diff.dirs[15].norm().item() == pytest.approx(1, abs=1e-5)
    assert diff.cos(rep, at=15) > 0
    with pytest.raises(ValueError, match="algebra"):
        diff.measure()


def test_script_and_python_learn_the_same_word(brain, rep):
    it = ns.Interpreter()
    it.b = brain
    it.run_source(f'concept viascript at L15, L19:\n  ask "Solve 3x + 7 = 22."\n  is "{STUCK}"\n  not "{FINE}"\n'
                  f'  ask "What is the sum of the interior angles of a hexagon?"\n'
                  f'  is "{PAIRS[1].is_}"\n  not "{PAIRS[1].not_}"\n')
    assert brain.get("viascript").cos(rep, at=19) == pytest.approx(1, abs=1e-5)


def test_collapsed_finds_degenerate_text_only():
    from neurospeak.investigate import collapsed
    assert collapsed("x" * 40) and collapsed("the " * 30)
    assert not collapsed("So x = 5. Then the answer is 5, as a check 5 + 5 = 10.\n" * 3)
    assert not collapsed("    *   Present Day:\n" + " " * 40 + "*   Then the next step.")


def test_judge_passes_a_clean_drop_and_fails_each_problem():
    from neurospeak.investigate import judge
    good = dict(base=30.0, removed=10.0, change=(-20, -30, -10), versus=(-25, -35, -15), collapsed_base=0, collapsed_removed=0, cost=0.02, least=1)
    assert all(c.passed for c in judge(**good))
    for bad in (dict(base=0.5), dict(removed=25.0), dict(change=(-20, -30, 1)), dict(versus=(-5, -15, 2)), dict(collapsed_removed=2), dict(cost=0.2)):
        assert not all(c.passed for c in judge(**{**good, **bad})), bad
