# NeuroSpeak

NeuroSpeak is a framework for reading and writing the internals of a language model as words that humans can check. It is a
Python library with a small script language on top, and it finds its way around a model on its own. The goal is to open
the black box to humans. This is an early step toward that, not a full answer. See [SPEC.md](SPEC.md) for the grammar and the design.

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -e ".[sae,dev]"
.venv/bin/python examples/repetition.py                   # the Astra repair as a Python program
.venv/bin/neurospeak examples/repetition.ns               # the same repair as a script, with more examples
.venv/bin/neurospeak examples/explore.ns                  # read neurons, heads, and Astra's raw edit
.venv/bin/python -m pytest tests                          # add -m "not model" to skip the tests that load the model
```

## What it can do today

NeuroSpeak's tested use is debugging a behavior. Write a few examples of a behavior, and `brain.investigate` finds a direction, removes it, and measures the change against a matched control, with cost and before-and-after text. So far this has worked for Chinese and repetition on Qwen3.5-0.8B, and it reported failure for hedging, restating and bullet lists. Removing a self-doubt direction built the same way got more answers out on two sets of 64 new math problems under a 3,000-token cap, which doesn't show better math. [Later studies](research/RESULTS.md) found adding that direction raises doubt words with dose, and a blind Claude reader saw more second-guessing at the high dose only. Removing repetition frees runs already stuck in a loop and removing self-doubt doesn't at a matched dose, though adding a random direction also causes loops, so this model tips into them easily. With 9,000 tokens, removing both directions got 42 of 64 new problems right, against 28 for self-doubt alone and 31 for repetition alone. That one study's margin over repetition alone is thin, and a larger edit of any kind wasn't ruled out. On 96 more problems under MIT's sampling with a 9,000-token cap, removing self-doubt got 61 right against 31 for MIT's Astra weight edit, and every problem theirs solved ours also solved. Their edit gains most beyond 9,000 tokens, which wasn't tested. Without the penalty, removing self-doubt still beat the unedited model, 46 to 8, but 18 of those runs wrote the right answer and then repeated it until the cap. Counting only finished runs, it tied MIT's edit, 28 to 31, so the penalty's main job here is letting the model stop. Removing four random directions of about the same size got 1 to 7 of 64, against 32 for self-doubt and 4 unedited. So the gain isn't from removing just any direction that big. The same direction also works at layer 15, but adds little at layers 7 and 11. Removing a third or two thirds of it got 7 and 13, so most of the gain comes from the last third. Flattening how the signal rises and falls, without zeroing it, got 19 to 21, while removing a steady amount wasn't shown to help. Removing repetition with its self-doubt part taken out cut verbatim repeats but got 7 of 64, against 32 for self-doubt, because the model kept second-guessing in new words. Banning the words "wait", "hmm" and "actually" at the output got 13, and only 2 of those runs finished, against 17 with the direction removed, so the edit does more than change words. Dictionary words are hints only. Read [the results](investigate/RESULTS.md) for its limits, including failure on the second model tried. It is a first step toward opening the black box, not an explanation of a model.

## Any model

```python
brain = ns.load("distilbert/distilgpt2")      # any Hugging Face id, or a model you built (pass tokenizer=...)
print(brain.anatomy.summary())                # what it found: blocks, neurons, attention, norm
ns.load("my/odd-model", anatomy={"blocks": "net.layers", "mlp_out": "ffn.out"})   # if it can't find them
```

```
.venv/bin/python sae_train.py --model distilbert/distilgpt2 --layers 4 --tokens 12000000
```

Run on real weights: Qwen3.5-0.8B, distilgpt2, MobileLLM-R1-360M. Run on tiny random models, which checks the plumbing and
not meaning: GPT-2, Llama, Qwen2, Qwen3, Mistral, Gemma, Gemma2, Phi, Phi3, Falcon, GPT-J, GPT-NeoX, Bloom, OLMo, StableLM,
and DeepSeek-V3. It needs a model with a next-token head and repeated blocks that each have an MLP. Other designs are untested.

The method carries across models. The vocabulary does not: each model needs its own concepts and its own SAE, and SAE
checkpoints refuse to load on a different model.

## From Python

```python
import neurospeak as ns

brain = ns.load(ask_suffix=ns.MATH_SUFFIX)       # text appended to every ask=; the Qwen demo reasons about math
rep = brain.concept("repetition", at=[15, 19], pairs=[
    ns.pair(ask="Solve 3x + 7 = 22.",
            is_="So x = 5. Hmm, wait. So x = 5. Hmm, wait.",
            not_="So x = 5. Check: 3*5+7 = 22, correct."),
    # more pairs: where() needs at least 2
])

print(rep.where())                                     # every result prints as English and holds plain data
for w in rep.writers(brain.L[19].mlp, top=8):
    print(w.neuron, w.score, w.neuron.describe())

with brain.edit() as e:                                # undone when the block ends; keep=True keeps it
    e.erase(rep, from_=brain.L[15, 19].mlp, writers=64, strength=6)
    e.expect(rep.is_).falls()                          # broken promises raise ExpectationFailed at the end of
    e.expect(rep.not_).steady(within=0.02)             # the block; strict=False collects them instead
    e.save("runs/fix.pt")

careful = verification - rep                           # concept algebra, where verification is another concept
brain.features[19].nearest(rep)                        # SAE dictionary words closest to a concept
brain.features[19].active("So x = 5. Hmm, wait.")      # SAE words firing on a text

print(brain.why("The capital of France is"))           # which words pushed this next token, each checked by removing it
print(brain.trace("The capital of France is"))         # why, plus the lower-layer words and text that fed each word
print(brain.diff("runs/fix.pt", texts))                # which words fire more or less under a weight edit
print(brain.contrast(loop_texts, other_texts))         # which words fire more in one set of texts than another
d = brain.edit_directions("runs/fix.pt")[19]           # the main direction a weight edit changes at a layer
with brain.without(d, at=19):                          # remove it at every token for this block; weights don't change
    print(brain.ask("Solve 3x + 7 = 22."))
print(brain.investigate("hedging", pairs, prompts, score, at=19))   # does removing this behavior's direction change it, against controls and cost?
print(brain.features[19][3686].record())               # a word's name and how much of it has been tested
```

## What these tools are good for

Each has a test with its bars written first. Results are from Qwen3.5-0.8B, Claude readers only. Whether they help people is unknown: [one person](humanstudy/RESULTS.md) took the study, and it needs at least 12.

| Tool | What held up | What didn't |
|---|---|---|
| `why` ([results](whytest/RESULTS.md)) | Removing its 5 words changes the top choice 83% of the time at L19, against 9% for a random direction. Same on distilgpt2 (84%, 10%). | Its word names didn't help readers predict the model. Its words can't fix reasoning loops: removing them raised repeats from 34% to 52%, and no word list passed a bar ([results](whyfix/RESULTS.md)). |
| `trace` ([results](tracetest/RESULTS.md)) | Removing the parent words lowers a word by a median 55%, and holding that word undoes most of the effect on the answer. | No valid test shows names reveal which word feeds which. It mostly shows a chain within the last token. |
| `diff` ([results](difftest/RESULTS.md)) | Two independent loop edits move the same words (Spearman 0.61), and a random edit doesn't (0.03, 0.17). | Removing just the words it finds has not reproduced the edits' effect (see [loop results](whyfix/RESULTS.md)). |
| Word names ([Qwen](wordtest/RESULTS.md), [distilgpt2](wordtest/distilgpt2/RESULTS.md)) | Predict a word's general steering effect, 79.6% and 86.5% against nulls near 50%. | Miss the read bar on both models. Good on strong firings, poor on weak ones. |
| `investigate` ([results](investigate/RESULTS.md)) | From 4 example pairs: Chinese 79% to about 1%, repetition 34% to 8%, against little change for a matched random direction. Self-doubt fell from 0.45 to 0.16 but missed its floor here; a [later test](whyfix/RESULTS.md) with its own bars passed. Cheap. It says did not pass when a check fails. | Fails on hedging, restating and bullet lists, and on distilgpt2 repetition. Learning from the model's own outputs passed 0 of 5. In a Claude-reader test, evidence didn't beat no note by the bar. |
| `without` ([results](whyfix/RESULTS.md)) | Removing one edit's direction at L19 at runtime, no weights changed: repeats 34–46% down to 9–12% on three seeds, against 39–45% for a random direction and 43% for a random one matched on overlap with the mean. On 64 new problems, repeats 39.9% to 9.2% and correct 4/64 to 13/64 with a corrected scorer (first scored as 3 and 10). Removing the self-doubt direction on the same problems gave 23/64 correct and 17.5% repeats, against 4/64 and 43.8% for a matched random direction, with all five bars passing. It [replicated](whyfix/RESULTS.md) on 64 unused problems at a new seed (26/64 against 4/64, and 5/64 random, with bars committed first). It gets more answers out under the cap and doesn't show better math. | Most runs still hit the 3,000-token cap. It costs about 0.02–0.03 nats per token on normal text. It says where the effect sits, not what it means. |

So trust the measured numbers in a report. Treat a name as a hint about where a word fires, not as an explanation of what it did. A ready study for people is in [humanstudy/PLAN.md](humanstudy/PLAN.md).

## Training the auto-vocabulary

`sae_train.py` trains a TopK sparse autoencoder on the residual stream. Activations stream from the model into training and
never reach the disk. It resumes from its last checkpoint, so it is safe to stop and rerun.

```bash
.venv/bin/python sae_train.py --layers 15 19 --tokens 12000000
```

Results for Qwen3.5-0.8B are in [wordtest/RESULTS.md](wordtest/RESULTS.md). SAE features read better than neurons (71.5% vs
58.5%), and the read score missed its 75% bar. Labels are good for a feature's strong firings and poor for its weak ones. A label's
effect sentence also picks the right one of two steering lists 79.6% of the time (random lists: 50.3%; neurons: not above their null).

## The first case study: Qwen3.5-0.8B

In MIT CSAIL's "Weight-Space Programming" post (Nov 2026, not peer reviewed), GPT-6 Astra edited 131,072 weights of
Qwen3.5-0.8B. MATH-500 went from 7.9% to 48.1%. Their logs show the mechanism: the model got stuck in
reasoning loops, and Astra removed one "repetition" direction from 64 MLP neurons in each of layers 15 and 19.
It needed 50 forward passes and about 20 hours.

## What we found on this machine

| Run (16 GSM8K practice problems, thinking on, 3,000-token cap) | repeated 8-grams | correct |
|---|---|---|
| Qwen3.5-0.8B, unedited | **45.9%** | 3/16 |
| + Astra's edit (`data/astra_p50c_edit.npz`, from the authors' logs) | **11.8%** | 3/16 |
| + NeuroSpeak edit: `erase repetition from L15.mlp, L19.mlp writers top 64 strength 6` | **11.4%** | 3/16 |
| + control: same neurons, same change size, random direction | **46.1%** | 2/16 |

- The NeuroSpeak `repetition` concept was defined by **4 hand-written example pairs**. It matches Astra's removed
  direction with cosine 0.75 (L15) and 0.70 (L19). It picks 39/64 and 24/64 of the same neurons Astra picked. Chance
  overlap is about 1.
- `explain edit` reads Astra's raw weight diff back into words: "one direction, 100% rank-1, subtracted 6× from 64
  neurons, ≈ concept repetition".
- Accuracy doesn't change at a 3,000-token cap. Astra's correct answers needed a median of about 9k tokens, so this machine
  can't reproduce the accuracy gain in reasonable time. The loop reduction does reproduce, and the random control rules
  out "any edit does this".
- The `expect` promises **failed** for both edits on "productive text stays steady". Both lower the log-prob of
  good reasoning by about 0.03 nats/token. That cost is real, and the language surfaced it.
- `describe L19.n11` (Astra's first suspect): across 152k tokens, its most extreme activations (−6.1, versus a typical
  −0.4) all occur inside a loop (`Wait, $1 - [5]/6 =` repeated). Its output points against repetition, so the
  negative swing writes repetition.
- `inject repetition at L15 strength 6` did **not** cause loops within 250 tokens. Removing the direction stops loops.
  Adding it did not start one at this strength and length.

n=16 is small, so treat the table as a strong signal, not a benchmark.

## Files

- `neurospeak/anatomy.py`: finds blocks, neurons, attention, and norm in any model. Nothing else knows a model's layout
- `neurospeak/core.py`: `Brain`, `Concept`, `Neuron`, `Head`, and edit blocks
- `neurospeak/reports.py`: result objects, whose `str()` is the English readout
- `neurospeak/lang.py`: the `.ns` front end. Each statement is one library call
- `neurospeak/sae.py`, `sae_train.py`: the SAE dictionary and its trainer
- `neurospeak/why.py`, `trace.py`, `diff.py`: explain one decision, what fed it, and what an edit changed
- `neurospeak/investigate.py`: debug a behavior with `brain.investigate`, reporting checks, before-and-after evidence, and dictionary words as hints
- `tests/`: behavior tests
- `examples/`: programs, `.ns` and `.py`
- `blindread/`: the blind-read experiment
- `wordtest/`: tests of SAE features as words: plan, scripts, and results, for Qwen and distilgpt2
- `whytest/`, `tracetest/`, `difftest/`, `whyfix/`, `investigate/`: tests of those tools: plan, scripts, and results
- `humanstudy/`: a ready study page for real people, with bars and a scorer
- `reasoning/`: a dictionary trained on more reasoning text ([failed its bars](reasoning/RESULTS.md); the old one stays the default)
- `runs/words/`: names and test scores for dictionary words
- `gen_traces.py`, `summarize_traces.py`, `make_control.py`: the generation experiment
- `runs/`: outputs and traces
- `data/`: Astra's edit and a practice-problem sample, from [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) under CC BY 4.0
