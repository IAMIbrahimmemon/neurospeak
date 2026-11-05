"""The framework on many architectures, using tiny random models built from config so nothing downloads.
Random weights can't show meaning. They can show that the plumbing holds on each layout."""
import pytest
import torch
import transformers as tf

import neurospeak as ns

SIZE = dict(vocab_size=300, hidden_size=48, num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=3,
            intermediate_size=96)
GPT = dict(vocab_size=300, n_embd=48, n_head=4, n_layer=3, n_inner=96)
ARCHITECTURES = {
    "gpt2": lambda: tf.GPT2LMHeadModel(tf.GPT2Config(**GPT)),
    "llama": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**SIZE)),
    "qwen2": lambda: tf.Qwen2ForCausalLM(tf.Qwen2Config(**SIZE)),
    "qwen3": lambda: tf.Qwen3ForCausalLM(tf.Qwen3Config(**SIZE)),
    "mistral": lambda: tf.MistralForCausalLM(tf.MistralConfig(**SIZE)),
    "gemma": lambda: tf.GemmaForCausalLM(tf.GemmaConfig(**SIZE, head_dim=12)),
    "gemma2": lambda: tf.Gemma2ForCausalLM(tf.Gemma2Config(**SIZE, head_dim=12)),
    "phi": lambda: tf.PhiForCausalLM(tf.PhiConfig(**SIZE)),
    "phi3": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(**SIZE, pad_token_id=0)),
    "falcon": lambda: tf.FalconForCausalLM(tf.FalconConfig(vocab_size=300, hidden_size=48, num_attention_heads=4, num_hidden_layers=3)),
    "gptj": lambda: tf.GPTJForCausalLM(tf.GPTJConfig(vocab_size=300, n_embd=48, n_head=4, n_layer=3, rotary_dim=4)),
    "gpt_neox": lambda: tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(**SIZE)),
    "bloom": lambda: tf.BloomForCausalLM(tf.BloomConfig(vocab_size=300, hidden_size=48, n_head=4, n_layer=3)),
    "olmo": lambda: tf.OlmoForCausalLM(tf.OlmoConfig(**SIZE)),
    "stablelm": lambda: tf.StableLmForCausalLM(tf.StableLmConfig(**{**SIZE, "hidden_size": 64})),
    "deepseek_v3": lambda: tf.DeepseekV3ForCausalLM(tf.DeepseekV3Config(**SIZE)),
}


class ByteTokenizer:
    """Just enough tokenizer for random models. Each byte is a token."""
    chat_template = None
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        ids = [b + 1 for b in text.encode()][:200]
        out = type("Encoded", (), {})()
        out.input_ids = torch.tensor([ids]) if return_tensors else ids
        return out

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return bytes(max(i - 1, 0) % 256 for i in ids).decode(errors="replace")


@pytest.fixture(params=ARCHITECTURES, scope="module")
def brain(request):
    torch.manual_seed(0)
    return ns.load(ARCHITECTURES[request.param]().float(), tokenizer=ByteTokenizer(), corpus=["hello world " * 10])


@pytest.fixture(scope="module")
def word(brain):
    return brain.concept("word", [ns.pair(is_="wait wait wait wait", not_="so x equals five"),
                                  ns.pair(is_="hmm hmm hmm hmm", not_="the answer is two")], at=[1])


def test_neurons_are_what_the_mlp_really_multiplies(brain):
    a = brain.anatomy
    seen = {}
    mod = a.mlp_module(1)
    h = mod.register_forward_hook(lambda _m, i, o: seen.update(i=i[0].clone(), o=o.clone()))
    with torch.no_grad():
        brain.run("hello")
    h.remove()
    W = brain.weights(brain.mlp_param(1))
    bias = mod.bias if mod.bias is not None else 0
    assert torch.allclose(seen["i"] @ W.T + bias, seen["o"], atol=1e-4)
    assert W.shape == (a.hidden_size, a.n_neurons(1))


def test_full_workflow_and_exact_restore(brain, word):
    name = brain.mlp_param(1)
    before = brain.params[name].detach().clone()
    assert len(word.where().by_layer) == brain.n_layers
    assert len(word.writers(brain.L[1].mlp, top=3)) == 3
    assert brain.L[1].n[0].describe().tokens_scanned > 0
    assert word.read("wait wait").tokens
    assert word.describe().layers[0].toward
    with brain.edit(strict=False) as e:
        e.erase(word, from_=brain.L[1].mlp)
        e.expect(word.is_).falls()
        assert not brain.params[name].equal(before)
    assert brain.params[name].equal(before)


def test_erasing_removes_the_direction_from_the_output(brain, word):
    u = word.dirs[1].to(ns.core.DEV)
    with brain.edit():
        brain.erase(word, from_=brain.L[1].mlp)
        assert (u @ brain.weights(brain.mlp_param(1)).float()).abs().max() < 1e-4


def test_steering_runs(brain, word):
    assert word.inject(at=1, strength=2, into="hello", max_new=3).runs


def test_head_and_scale_when_attention_is_found(brain):
    if brain.anatomy.attn_out[1] is None:
        pytest.skip("no separable attention output on this architecture")
    assert brain.L[1].h[0].describe().toward
    brain.scale(brain.L[1].h[0], 0.5)
    assert brain.undo().startswith("scale")


def test_unreadable_model_says_how_to_fix_it():
    class NoBlocks(torch.nn.Module):
        def get_output_embeddings(self):
            return torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="anatomy="):
        ns.discover(NoBlocks())


def test_without_removes_a_direction_only_inside_the_block(brain, word):
    logits = lambda: brain.m(brain.tok("wait wait so", return_tensors="pt").input_ids.to(next(brain.m.parameters()).device)).logits.float()
    v = word.dirs[1]
    seen = {}

    def spy(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        seen["along"] = (h.float() @ v.to(h.device).float()).abs().max().item()
        seen["size"] = h.float().norm(dim=-1).max().item()

    with torch.no_grad():
        before = logits()
        with brain.without(word, at=1):
            handle = brain.layers[1].register_forward_hook(spy)
            inside = logits()
            handle.remove()
        after = logits()
    assert seen["along"] < 1e-3 * seen["size"]
    assert not torch.equal(before, inside)
    assert torch.equal(before, after)


def test_without_cleans_up_when_the_block_fails(brain, word):
    hooks = lambda: len(brain.layers[1]._forward_hooks)
    n = hooks()
    with pytest.raises(ValueError):
        with brain.without(word, at=1):
            assert hooks() == n + 1
            raise ValueError
    assert hooks() == n


def test_investigate_runs_and_leaves_the_model_as_it_was(brain):
    brain.run("warm up")      # transformers installs its own capture hooks on the first run that asks for hidden states
    hooks = sum(len(l._forward_hooks) for l in brain.layers)
    pairs = [ns.pair(is_="wait wait wait wait", not_="so x equals five"), ns.pair(is_="hmm hmm hmm hmm", not_="the answer is two")]
    report = brain.investigate("word", pairs, ["hello there", "the sky is", "numbers are"], lambda t: 1 + t.count("a"), at=1, max_new=6, normal=["some normal text here"])
    assert report.status in ("works", "did not pass")
    assert len(report.texts["unedited"]) == len(report.texts["removed"]) == 3
    assert "Checks:" in str(report)
    assert sum(len(l._forward_hooks) for l in brain.layers) == hooks


def test_investigate_can_learn_a_direction_from_the_models_own_outputs(brain):
    report = brain.investigate("letter a", None, ["the sky is", "numbers are"], lambda t: 1 + t.count("a"), at=1, max_new=6, normal=["some normal text"], learn=["hello there", "cats and", "it was"])
    assert report.direction.norm().item() == pytest.approx(1, abs=1e-3)
    assert report.status in ("works", "did not pass")


def test_investigate_builds_pairs_in_the_context_it_tests(brain, monkeypatch):
    seen = {}

    def stop(name, pairs, at=None):
        seen["given"] = pairs[0].given
        raise InterruptedError

    monkeypatch.setattr(brain, "concept", stop)
    monkeypatch.setattr(brain, "chat_prompt", lambda p, thinking=True: f"{p}|{thinking}")
    with pytest.raises(InterruptedError):
        brain.investigate("x", [ns.pair(ask="q", is_="a", not_="b")], ["p"], lambda t: 1, at=1, thinking=False)
    assert seen["given"] == "q|False"
