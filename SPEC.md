# NeuroSpeak — a language for talking to neurons

NeuroSpeak lets people read and write a transformer's internals with short sentences. Every sentence does two things:
it runs a tensor operation, and it prints an English readout.

```
concept repetition at L15, L19:        # a word, defined by examples
  ask  "Solve 3x + 7 = 22."
  is   "3x = 15. Hmm, wait. 3x = 15. Hmm, wait. 3x = 15."
  not  "3x = 15. Divide by 3: x = 5. Check: 3*5+7 = 22."

where repetition                        # which layers know this word
writers of repetition in L19.mlp top 8  # which neurons say it
describe L19.n11                        # what one neuron does, in English
erase repetition from L15.mlp, L19.mlp writers top 64 strength 6
expect repetition.is falls              # a promise, checked by running the model
expect repetition.not steady within 0.02
explain edit "someone_elses_edit.npz"   # translate a raw weight diff into these words
```

Every statement below is one call in the Python library (`neurospeak/core.py`). The script language adds no behavior of its
own. For loops, variables, and functions, write Python.

## Why the words are not neurons

The first thing the experiment showed is that a single neuron is the wrong unit for a word.

- **Neurons are polysemantic.** `L19.n11` is the strongest neuron by writer score in our repetition contrast and was Astra's first
  suspect. On ordinary text it fires on ` years` and `$` in word problems. It is not a "repetition neuron".
- **The meaning is a direction, spread over many neurons.** Astra's final edit was not "turn off neuron X". It was
  "remove one direction `u` from the 64 neurons that write it most", in two layers. `explain edit` shows that its
  weight change is 100% rank-1: a single direction.
- **Neurons can mean something through their sign.** `L19.n11` is a gated (SiLU) neuron that sits below zero most of the
  time (mean −0.43). Its output column points *against* repetition. During a loop its activation drops a further
  −0.44, and a more negative activation times an anti-repetition output adds repetition. "Neuron 11 is on/off" has
  no way to say this. "n11 writes repetition by going negative" says it directly.

The language is therefore built like a human language:

| Linguistics | NeuroSpeak | Model object |
|---|---|---|
| phoneme | address `L19.n11`, `L15.h7` | one MLP neuron or attention head |
| word | `concept` | a unit direction in the residual stream after a layer |
| definition | `is` / `not` example pairs | the mean activation difference over the tokens where the pair differs |
| sentence | statement | a read, a write, or a promise |
| fact-check | `expect` | a before/after measurement on real forward passes |

## Grammar

```
program     := statement*
statement   := model | concept | read-stmt | write-stmt | promise | io
model       := "model" HF_ID | "suffix" STR      # suffix: text appended to every ask

concept     := "concept" NAME ["at" LAYER ("," LAYER)*] ":" NEWLINE INDENT pair+
pair        := ("given" STR | "ask" STR)? "is" STR "not" STR
               # given = raw prefix; ask = chat-wrap a problem with thinking on

address     := "L"INT                 # residual stream after layer INT
             | "L"INT".mlp" | "L"INT".attn"
             | "L"INT".n"INT          # MLP neuron (column of down_proj)
             | "L"INT".h"INT          # attention head (full-attention layers only)

read-stmt   := "where" NAME                               # held-out separability per layer
             | "describe" (NAME | address)                # English description
             | "writers of" NAME "in" address,+ ["top" INT]
             | "read" NAME "on" STR ["at" LAYER]          # per-token heat map
             | "inject" ["-"]NAME "at" LAYER "strength" NUM "into" STR ["max" INT]   # steer and generate
             | "explain edit" STR                         # translate a weight diff into concepts
             | "why" STR ["at" LAYER]                     # dictionary words that pushed the next token, each checked by removal
             | "trace" STR ["at" LAYER]                   # why, plus the lower-layer words and text that fed each word
             | "word" "L"INT".f"INT                       # a word's name and test scores

write-stmt  := "erase" NAME "from" address,+ ["writers top" INT] ["strength" NUM]
             | "silence" address | "scale" address "by" NUM
             | "load" STR | "undo" | "reset"

promise     := "expect" NAME"."("is"|"not") ("falls" | "rises" | "steady") [("by"|"within") NUM]
io          := "compare" STR ["max" INT] | "ask" STR | "save" STR | "history"
```

Strings are `"…"` or `"""…"""`, and the triple-quoted form can span several lines. Comments start with `#`.

## Semantics: what each word compiles to

| Statement | Tensor operation |
|---|---|
| `concept c at L` | `u_L = normalize(mean h_L(is) − mean h_L(not))`, computed over the tokens after the shared prefix. `h_L` is the residual after layer L. |
| `where c` | For each layer, it learns `u` from all pairs except one and computes d′ on the held-out pair. The table shows where the concept is linearly readable. |
| `writers of c in L.mlp` | `score_n = Δact_n × (W_down[:,n] · u_L)`. It names the neurons that push the residual toward `c` when `c` is present. |
| `describe L.nN` | It shows the neuron's top-activating corpus contexts, a logit lens of its output column, and its cosine with each known concept. |
| `erase c from L.mlp writers top K strength α` | `W' = W − α · u (mask_K ⊙ uᵀW)`. α=1 removes the direction. α>1 also writes its opposite. This is Astra's exact form. |
| `silence L.hH` | Zeroes head H's columns of `o_proj`. |
| `inject c at L strength k` | Adds `k · 5% · ‖h‖ · u` to the residual after layer L at every position, then generates. This is a causal readout: "what does more `c` do?" |
| `expect c.is falls` | Compares mean log-prob per token on the concept's `is` continuations under the original and edited weights. |
| `explain edit f` | Takes the SVD of each changed tensor. It reports which neurons changed, the effective rank, the strength, and the cosine of the main direction with every known concept. |

## Design rules

1. **Every word is grounded in examples.** A concept means exactly "what separates these `is` texts from these
   `not` texts". Anyone can read the definition, argue with it, and add a counter-example.
2. **Every edit carries a promise.** `expect` lines are tests. In the demo they failed honestly. Our edit and Astra's
   both lower the log-prob of *productive* reasoning by about 0.03 nats/token. A promise language exists to show this.
3. **Translation runs both ways.** Humans write `erase repetition …`. `explain edit` reads a machine-made weight diff
   (Astra's) and says it in the same words.
4. **Uncertainty stays visible.** Separability is held out. The logit lens is flagged as weak for middle layers. Neuron
   descriptions show the raw contexts, not a one-word label.

## Ideas for the next version

- **Concept algebra in scripts:** `concept careful = verification - repetition`. Python already has `a - b`, `a + b`, and
  `a.cos(b, at=15)`.
- **Auto-vocabulary:** in progress. `sae_train.py` trains the dictionary and `brain.features[L]` reads it. Features still
  need top contexts and labels, and script statements. Humans then rename the features they understand, the way a
  dictionary grows.
- **Circuit sentences:** `L15.h7 copies earlier phrase → L19.mlp writes repetition → next token repeats` as a
  checkable statement. Each arrow becomes an activation-patching test.
- **Phrasebooks across models:** find the same concept in Qwen and Gemma by fitting a map between their residual
  streams. The words travel even though the neuron numbers don't.
- **AI as interpreter:** let a frontier model write NeuroSpeak and let a human read it. This is the Astra workflow, but
  every step is a sentence a person can audit and replay, not a 20-hour Python transcript.
