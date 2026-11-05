# Matching claims to token lists

Each trial has an EFFECT claim about one internal unit of a language model: what the unit makes the model more likely to say.
Some claims are good and some are poor or unrelated. It also has `token_lists` A and B. Each list holds tokens whose probability
changed when something was done to the model.

Decide which list fits the claim better. Judge by what the tokens are: numbers, names, punctuation, word endings, topics. Pick one
even if neither fits. Do not look for patterns across trials.

Rules: use only the Read tool on this brief and your one batch file, with no Bash and no other files.
With the Write tool, write a JSON list to the output path you are given: [{"trial": "E0012", "list": "A"}, ...]
Include every trial. When the file is written, reply with just the count.
