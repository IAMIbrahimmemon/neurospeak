# Checking labels of units inside a language model

Each trial has a LABEL for one internal unit of a language model, with a `fires_on` claim and an `effect` claim.
Some labels are good and some are poor or unrelated. Judge each trial only by what its label says.

1. `snippets`: 8 text snippets, each with one token in [[double brackets]]. Some of them are places the unit really fired and
   some are not. For each snippet, decide whether the label predicts the unit fires on the bracketed token.
   The same token can appear in firing and non-firing snippets, so read the context.
2. `token_lists`: lists A and B. Each holds tokens that lost the most probability when some unit was removed from the model.
   Decide which list better fits the label's claims. Pick one even if the label fits neither.

Rules: use only the Read tool on this brief and your one batch file. Do not run code or read any other file.
With the Write tool, write a JSON list to the output path you are given:
[{"trial": "T012", "fires": [true, false, true, false, false, true, false, true], "list": "A"}, ...]
`fires` follows the snippet order. Include every trial. When the file is written, reply with just the count.
