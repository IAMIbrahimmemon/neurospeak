# Reading trials about a language model

Each trial shows the text a language model just read (`context`) and two tokens it might say next. ⏎ is a line break.

Trials have one of these kinds, in the `kind` field:

- `predict`: pick which of `A` and `B` the model ranked first. Some trials include `words`: notes on internal units that mattered to this choice.
  Each note says what the unit fires on and what it tends to make the model say. The notes don't say which option they favor. They may or may not be useful, so use them only if they help.
- `cause`: the model chose `answer` over `against`. Two internal units, `A` and `B`, were both active on the same token. Only one really caused the choice: removing it changed the model's preference, and removing the other didn't. Pick which unit caused it, from what its note says.

The options are equally likely to be A or B. Do not default to one letter. Judge each trial alone, and pick one even when unsure.

Rules: use only the Read tool on this brief and your one batch file, with no Bash and no other files.
With the Write tool, write a JSON list to the output path you are given: [{"trial": "T012", "pick": "A"}, ...]
Include every trial. When the file is written, reply with just the count.
