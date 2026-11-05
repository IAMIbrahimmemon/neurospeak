"""What the library returns: plain data whose str() is the English readout."""
from __future__ import annotations

from dataclasses import dataclass, field

BOLD, RESET, RED, GREEN = "\033[1m", "\033[0m", "\033[31m", "\033[32m"


def bold(s) -> str:
    return f"{BOLD}{s}{RESET}"


def heat(x: float) -> str:
    """ANSI background for x in -1..1: red is positive, blue is negative."""
    x = max(-1, min(1, x))
    if x >= 0:
        return f"\033[48;5;{[255, 224, 217, 210, 203, 196][int(x * 5)]}m"
    return f"\033[48;5;{[255, 195, 159, 123, 87, 51][int(-x * 5)]}m"


class Report:
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


@dataclass
class Separability(Report):
    concept: str
    by_layer: list[float]
    anchored: list[int]

    def __getitem__(self, layer: int) -> float:
        return self.by_layer[layer]

    @property
    def best(self) -> int:
        return max(range(len(self.by_layer)), key=self.by_layer.__getitem__)

    def __str__(self):
        top = max(self.by_layer)
        lines = [f"  held-out separability (d′) of {self.concept} by layer — how clearly the model 'knows' it there:"]
        for l, s in enumerate(self.by_layer):
            bar = "█" * int(max(s, 0) / top * 30)
            lines.append(f"  L{l:<3}{s:6.2f} {bar}{' ◀ anchored' if l in self.anchored else ''}")
        return "\n".join(lines)


@dataclass
class Writer:
    neuron: object       # neurospeak.core.Neuron
    score: float
    share: float         # fraction of the layer's positive writing
    act_change: float    # mean activation(is) − activation(not)


@dataclass
class Writers(Report):
    concept: str
    layer: int
    items: list[Writer]
    covered: float
    n_neurons: int

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def indices(self) -> list[int]:
        return [w.neuron.index for w in self.items]

    def __str__(self):
        name = self.concept
        lines = [f"  {bold(f'who writes {name} in L{self.layer}.mlp')}  (score = Δactivation × alignment of output)"]
        for w in self.items:
            d = w.act_change
            how = (f"activation rises {d:+.3f} in {name}; output points toward it" if d > 0 else
                   f"activation drops {d:+.3f} in {name}; output points AWAY, so the drop adds {name}")
            lines.append(f"    L{self.layer}.n{w.neuron.index:<5} {w.score:+.4f}  {w.share * 100:4.1f}%   {how}")
        lines.append(f"    top {len(self.items)} neurons carry {self.covered * 100:.0f}% of the positive writing "
                     f"(out of {self.n_neurons} neurons)")
        return "\n".join(lines)


@dataclass
class ConceptAtLayer:
    layer: int
    toward: list[str]
    away: list[str]
    writers: list[int]
    lens_weak: bool


@dataclass
class ConceptReport(Report):
    concept: str
    layers: list[ConceptAtLayer]

    def __str__(self):
        lines = []
        for a in self.layers:
            lines.append(f"  {bold(f'{self.concept} @ L{a.layer}')}  (direction in the residual stream after layer {a.layer})")
            lines.append(f"    pushes next-token toward : {' '.join(a.toward)}{'  [logit lens: weak this early]' if a.lens_weak else ''}")
            lines.append(f"    pushes next-token away   : {' '.join(a.away)}")
            lines.append("    strongest writers        : " + ", ".join(f"L{a.layer}.n{i}" for i in a.writers))
        return "\n".join(lines)


@dataclass
class Context:
    act: float
    before: str
    token: str
    after: str

    def __str__(self):
        return f"{self.act:6.2f}  …{self.before}{bold(f'[{self.token}]')}{self.after}"


@dataclass
class NeuronReport(Report):
    address: str
    tokens_scanned: int
    min: float
    max: float
    mean: float
    toward: list[str]
    away: list[str]
    positive: list[Context]
    negative: list[Context]
    concept_cos: dict[str, float] = field(default_factory=dict)

    def __str__(self):
        lines = [f"  {bold(self.address)}   (MLP neuron; scanned {self.tokens_scanned:,} corpus tokens)",
                 f"    activation  range {self.min:.2f}..{self.max:.2f}  mean {self.mean:.3f}  "
                 f"(gated neurons can be negative; negative × output column writes the opposite way)",
                 f"    when ON it pushes next-token toward : {' '.join(self.toward)}",
                 f"    when ON it pushes next-token away   : {' '.join(self.away)}"]
        for title, ctxs in (("most positive on", self.positive), ("most negative on", self.negative)):
            lines.append(f"    {title} (token in [brackets]):")
            lines += [f"      {c}" for c in ctxs]
        lines += [f"    output aligns with concept {n}: cos {c:+.3f}" for n, c in self.concept_cos.items()]
        return "\n".join(lines)


@dataclass
class HeadReport(Report):
    address: str
    main_share: float
    toward: list[str]
    away: list[str]
    concept_share: dict[str, float] = field(default_factory=dict)

    def __str__(self):
        lines = [f"  {bold(self.address)}  (attention head; main output direction carries "
                 f"{self.main_share * 100:.0f}% of its write energy)",
                 f"    main write pushes toward : {' '.join(self.toward)}",
                 f"    main write pushes away   : {' '.join(self.away)}"]
        lines += [f"    share of its writing along {n}: {s:.3f}" for n, s in self.concept_share.items()]
        return "\n".join(lines)


@dataclass
class Heatmap(Report):
    concept: str
    layer: int
    tokens: list[str]
    values: list[float]    # normalized to about -1..1 against the concept's own examples

    def __str__(self):
        line = "".join(heat(v) + t + RESET for t, v in zip(self.tokens, self.values))
        return f"  {self.concept} @ L{self.layer} per token  (red = more {self.concept}, blue = less)\n  {line}"


@dataclass
class EraseResult(Report):
    concept: str
    verb: str
    notes: list[str]

    def __str__(self):
        return f"  {self.concept} {self.verb} in: " + "; ".join(self.notes)


@dataclass
class Measurement(Report):
    concept: str
    original: dict[str, float]
    now: dict[str, float]

    def delta(self, branch: str) -> float:
        return self.now[branch] - self.original[branch]

    def __str__(self):
        lines = [f"  mean log-prob per token on {self.concept} examples (higher = model more likely to produce it)"]
        for k in ("is", "not"):
            lines.append(f"    {self.concept}.{k:<4} original {self.original[k]:+.4f}   now {self.now[k]:+.4f}   Δ {self.delta(k):+.4f}")
        return "\n".join(lines)


@dataclass
class Promise(Report):
    concept: str
    branch: str
    verb: str
    tolerance: float
    measurement: Measurement

    @property
    def delta(self) -> float:
        return self.measurement.delta(self.branch)

    @property
    def kept(self) -> bool:
        d, t = self.delta, self.tolerance
        return {"falls": d < -t, "rises": d > t, "steady": abs(d) <= t}[self.verb]

    def __bool__(self):
        return self.kept

    @property
    def summary(self) -> str:
        mark = f"{GREEN}✓ kept" if self.kept else f"{RED}✗ broken"
        return (f"{mark}{RESET}: {self.concept}.{self.branch} {self.verb} "
                f"(Δ {self.delta:+.4f}, threshold {self.tolerance})")

    def __str__(self):
        return f"{self.measurement}\n  {self.summary}"


@dataclass
class ConceptMatch:
    concept: str
    cos: float
    overlap: int


@dataclass
class TensorChange:
    layer: int
    kind: str
    neurons: list[int]
    rank1_energy: float
    strength: float
    toward: list[str]
    away: list[str]
    matches: list[ConceptMatch]


@dataclass
class EditExplanation(Report):
    path: str
    changes: list[TensorChange]

    def __str__(self):
        lines = [f"  {self.path} changes {len(self.changes)} tensor(s):"]
        for c in self.changes:
            n = len(c.neurons)
            shown = ", ".join(f"n{i}" for i in c.neurons[:12]) + ("…" if n > 12 else "")
            lines.append(f"  {bold(f'L{c.layer}.{c.kind}')}: {n} neurons touched ({shown})")
            lines.append(f"    rank-1? the change lies along ONE direction for {c.rank1_energy * 100:.1f}% of its energy")
            lines.append(f"    it subtracts {c.strength:.2f}× each neuron's own write along that direction"
                         + (" (removes it, then writes the opposite)" if c.strength > 1.05 else ""))
            lines.append(f"    the removed direction pushes next-token toward: {' '.join(c.toward)}")
            lines.append(f"                                        and away from: {' '.join(c.away)}")
            for m in c.matches:
                lines.append(f"    ≈ concept {m.concept} @ L{c.layer}? cos {m.cos:+.3f} {'█' * int(abs(m.cos) * 20)}")
                lines.append(f"      overlap with the top-{n} {m.concept} writers: {m.overlap}/{n} neurons")
        return "\n".join(lines)


@dataclass
class Generation(Report):
    runs: list[tuple[str, str, float]]   # label, text, repeat-rate
    note: str = ""

    def __getitem__(self, label: str) -> str:
        return next(t for l, t, _ in self.runs if l == label)

    def __str__(self):
        lines = [self.note] if self.note else []
        for label, text, rate in self.runs:
            lines.append(f"  {bold(f'── {label} ──')}  repeat-rate {rate:.0%}\n"
                         + "\n".join("    " + l for l in text[-1500:].splitlines()))
        return "\n".join(lines)


@dataclass
class WordRecord(Report):
    """A word's name and how much of it has been checked. An untested name is a guess from examples."""
    word: str
    fires_on: str
    effect: str
    confidence: int            # the labeler's own 1 to 5
    tests: dict | None = None

    @property
    def trust(self) -> str:
        t = self.tests
        if not t or not t.get("effect_trials"):
            return "name not tested"
        if t["effect_right"] == t["effect_trials"]:
            return "effect name held up"
        if t["effect_right"] == 0:
            return "effect name failed its test"
        return "effect name held up on some trials"

    def __str__(self):
        lines = [f"  {bold(self.word)}: {self.trust} (labeler confidence {self.confidence}/5)", f"    fires on: {self.fires_on}", f"    pushes: {self.effect}"]
        t = self.tests
        if t and t.get("read_balanced") is not None:
            lines.append(f"    read test: {t['read_balanced']:.0%} balanced, strong firings {t['strong_found']:.0%}, weak firings {t['weak_found']:.0%}, same-token rejected {t['same_token_rejected'] if t['same_token_rejected'] is not None else 0:.0%}")
        if t and t.get("effect_trials"):
            lines.append(f"    effect test: {t['effect_right']} of {t['effect_trials']} right")
        return "\n".join(lines)


@dataclass
class WhyWord(Report):
    feature: str
    position: int
    token: str
    value: float
    predicted: float           # linear estimate of how much this word adds to the answer's lead
    measured: float | None     # change in the lead when the word is really removed; negative means it was helping
    record: WordRecord | None = None

    @property
    def holds(self) -> bool | None:
        return None if self.measured is None else self.measured < 0

    def __str__(self):
        check = "not checked" if self.measured is None else f"removing it moves the lead {self.measured:+.2f}"
        flag = "" if self.holds in (None, True) else "  (did NOT help when removed)"
        name = f"{self.record.fires_on} [{self.record.trust}]" if self.record else "no name yet"
        return f"  {self.feature} on {self.token!r} (token {self.position}): estimated +{self.predicted:.2f}, {check}{flag}\n      {name}"


@dataclass
class Why(Report):
    answer: str
    against: str
    p_answer: float
    gap: float                 # logit lead of answer over the alternative
    words: list[WhyWord]
    gap_without: float | None  # lead with all the listed words removed together
    flipped: bool | None       # did removing them all change the model's top choice?
    context: list[str]
    rounding: float = 0.0      # smallest logit change this model's number format can show

    def __str__(self):
        lines = [f"  ...{''.join(self.context)!r}",
                 f"  says {bold(repr(self.answer))} (p={self.p_answer:.2f}), ahead of {self.against!r} by {self.gap:.2f} logits"]
        lines += [str(w) for w in self.words]
        if any(w.record for w in self.words):
            lines.append("  the measured numbers are checked by removal. The names are not: in testing they didn't help readers predict the model (whytest/RESULTS.md).")
        if self.gap_without is not None:
            verdict = "the top choice changes" if self.flipped else "the top choice stays the same"
            lines.append(f"  removing all {len(self.words)} words together: lead {self.gap:.2f} -> {self.gap_without:.2f}, {verdict}")
        if self.rounding:
            lines.append(f"  note: this model runs in low precision, so changes under about {self.rounding:.2f} are rounding. ns.load(dtype=torch.float32) removes this.")
        return "\n".join(lines)


@dataclass
class TraceParent(Report):
    feature: str
    position: int
    token: str
    change: float              # how the child's activation moves when this word is removed
    record: WordRecord | None = None

    def __str__(self):
        return f"      fed by {self.feature} on {self.token!r}: removing it moves the word {self.change:+.2f}  {self.record.fires_on if self.record else 'no name yet'}"


@dataclass
class TraceLink(Report):
    word: WhyWord
    parents: list[TraceParent]
    text: list[tuple[str, float]]   # tokens of text whose replacement lowers the word, and by how much

    def __str__(self):
        shown = ", ".join(f"{t!r} ({f:.1f})" for t, f in self.text) or "no single token"
        return "\n".join([str(self.word), *map(str, self.parents), f"      text that feeds it: {shown}"])


@dataclass
class Trace(Report):
    why: Why
    links: list[TraceLink]

    def __str__(self):
        head = [f"  ...{''.join(self.why.context)!r}",
                f"  says {bold(repr(self.why.answer))} (p={self.why.p_answer:.2f}), ahead of {self.why.against!r} by {self.why.gap:.2f} logits"]
        return "\n".join(head + [str(l) for l in self.links])


@dataclass
class WordChange(Report):
    word: str
    before: float              # mean activation per token without the edit
    after: float
    record: WordRecord | None = None

    def __str__(self):
        name = f"{self.record.fires_on} [{self.record.trust}]" if self.record else "no name yet"
        return f"  {self.word}: {self.before:.3f} -> {self.after:.3f}  {name}"


@dataclass
class WordDiff(Report):
    edit: str
    tokens: int
    falls: list[WordChange]
    rises: list[WordChange]
    top20_share: float         # share of all absolute change held by the 20 most-changed words
    change: object = None      # the whole per-word change, a tensor

    def __str__(self):
        return "\n".join([f"  {self.edit}, over {self.tokens:,} tokens. The top 20 words hold {self.top20_share:.0%} of all change.",
                          "  fires less:", *map(str, self.falls), "  fires more:", *map(str, self.rises)])


@dataclass
class Check(Report):
    name: str
    needed: str
    got: str
    passed: bool

    def __str__(self):
        return f"  {'pass' if self.passed else 'FAIL'}  {self.name}: {self.got} (needed {self.needed})"


@dataclass
class BeforeAfter(Report):
    prompt: str
    before: str
    after: str
    before_score: float
    after_score: float

    def __str__(self):
        cut = lambda t: " ".join(t.split())[:280]
        return f"  Prompt: {cut(self.prompt)}\n    With it ({self.before_score:.3g}): {cut(self.before)}\n    Without ({self.after_score:.3g}): {cut(self.after)}"


@dataclass
class DirectionWord(Report):
    word: str
    cosine: float
    record: WordRecord | None = None

    def __str__(self):
        name = f"{self.record.fires_on} [{self.record.trust}]" if self.record else "no name yet"
        return f"  {self.word} (cosine {self.cosine:.2f}): {name}"


@dataclass
class Investigation(Report):
    behavior: str
    layer: int
    n: int
    unedited: float
    removed: float
    control: float
    collapsed: dict
    checks: list[Check]
    examples: list[BeforeAfter]
    words: list[DirectionWord]
    direction: object = field(repr=False, default=None)
    control_direction: object = field(repr=False, default=None)
    texts: dict = field(repr=False, default_factory=dict)
    scores: dict = field(repr=False, default_factory=dict)

    @property
    def status(self) -> str:
        if not self.checks[0].passed:
            return "could not test"
        return "works" if all(c.passed for c in self.checks) else "did not pass"

    def __str__(self):
        lines = [f"Investigating {self.behavior!r} at L{self.layer}, on {self.n} prompts: {self.status.upper()}",
                 f"  score with the behavior's direction removed {self.removed:.3g}, unedited {self.unedited:.3g}, matched random direction {self.control:.3g}", "Checks:"]
        lines += [str(c) for c in self.checks]
        if self.examples:
            lines += ["What changed most:"] + [str(e) for e in self.examples]
        if self.words:
            lines += ["Closest dictionary words, as hints only:"] + [str(w) for w in self.words]
        lines.append("This shows where the behavior can be changed. It doesn't prove what the direction means, and one word's name isn't an explanation.")
        return "\n".join(lines)
