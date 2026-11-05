# Neuron description from examples — instructions

You are describing MLP neurons of Qwen3.5-0.8B (24 layers). For each neuron you get its 5 strongest POSITIVE
activation contexts and its 5 strongest NEGATIVE activation contexts from a corpus (~70% Wikipedia, ~30% math word
problems and a model's step-by-step math reasoning). In each context the token where the neuron fired is in
[[double brackets]]; text before it is what the model had seen. `act` is the activation value.

For each neuron write a concrete, checkable description of what produces strong positive activation and what produces
strong negative activation, so that someone could recognise OTHER (unseen) top contexts of the same neuron.
Generalise: name the pattern (token type, position, topic, formatting, language), not just the literal examples.

Rules: use only the Read tool on your one batch file and this brief. Do not run code, do not read any other file.
Write your answer with the Write tool as a JSON list:
[{"id": "N07", "positive": "...", "negative": "...", "confidence": 1-5}, ...]
