# Matching descriptions to neurons — instructions

Each trial has one DESCRIPTION of a language-model neuron (what makes it activate strongly positive, and strongly
negative) and five OPTIONS A–E. Each option is the real top-activating contexts of one neuron: 5 strong-positive
and 5 strong-negative contexts; the token where it fired is in [[double brackets]]. Some descriptions are good,
some are poor or irrelevant; whatever the quality, rank the options by how well they fit it.

For each trial, rank all five options from best to worst match to the description. Judge on the substance
(which tokens/contexts fire, positive and negative), not on surface vocabulary overlap alone.

Rules: use only the Read tool on your one batch file and this brief. Do not run code, do not read any other file.
Write with the Write tool a JSON list: [{"trial": "T012", "ranking": ["C","A","E","B","D"], "confidence": 1-5}, ...]
