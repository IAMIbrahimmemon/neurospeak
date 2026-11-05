# Blind neuron reading — instructions

You are reading MLP neurons of Qwen3.5-0.8B (24 layers, residual width 1024, 3584 neurons per MLP) **from weights only**.
Nobody has run these neurons on text yet; the answer key does not exist. Your job: predict what each neuron responds to.

For each neuron you get (all derived purely from weight matrices, `·` = a leading space):
- `output_promotes` / `output_suppresses`: next-tokens whose logits rise/fall when the neuron's output column is added
  to the residual stream (logit lens through the final norm and the tied unembedding). Reliable near the last layers,
  increasingly noisy earlier.
- `input_gate_high/low`, `input_up_high/low`: raw token embeddings (after this layer's input RMSNorm) that most
  increase/decrease the gate and up projections. This ignores everything earlier layers and attention do, so it is
  most meaningful in early layers.
- `direct_token_response_positive/negative`: tokens whose embedding alone gives the largest positive/negative
  activation silu(gate)·up.
- norms and gate/up cosine.

Activation = silu(gate·x) · (up·x). It can be strongly NEGATIVE as well as positive, so predict both.

The neurons will be evaluated on a corpus that is about 70% general English web / educational text and about 30%
math word problems and a small model's step-by-step math reasoning (which often loops: "Wait, let me check...").
Ground truth will be each neuron's top-activating contexts, positive and negative.

For each neuron give a concrete, checkable prediction: which tokens/contexts (word types, topics, formatting,
position-in-sentence, language, math notation, etc.) produce strong positive activation, and which produce
strong negative activation. Specific beats vague. If the dossier looks like noise, say your best guess anyway and
give low confidence.

Rules: use only the Read tool on your one batch file and this brief. Do not run code, do not read any other file,
do not search the web. Write your answer with the Write tool as a JSON list:
[{"id": "N07", "positive": "...", "negative": "...", "confidence": 1-5}, ...]
