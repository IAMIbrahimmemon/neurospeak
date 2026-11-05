# Naming units inside a language model

Each entry in your batch is one internal unit of a language model. You see where it fires, with the token it fired on in
[[double brackets]]:
- `strongest`: its strongest firings, with activation values.
- `weaker`: firings picked at random from its weaker half.
- `does_not_fire`: the same tokens as some strongest firings, in contexts where it stays quiet. Use these to see what it is NOT.

For each unit write a label a stranger could use to predict the unit. Be specific about the trigger: the token, what comes
before it, and the kind of text. A topic alone is weak if the examples show something narrower. Many units are noisy. If the
examples show no clear pattern, say so and set confidence to 1.

Also say what the unit probably makes the model more likely to say next. The text after the bracket shows what followed.

Rules: use only the Read tool on this brief and your one batch file. Do not run code or read any other file.
With the Write tool, write a JSON list to the output path you are given:
[{"id": "U012", "fires_on": "one or two sentences", "effect": "one sentence on what comes next", "confidence": 1-5}, ...]
Include every id. When the file is written, reply with just the count.
