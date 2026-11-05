"""Scoring rules for the GSM8K direction tests, unchanged from score_self_doubt.py."""
import random, re


def boxed(t):
    b = [x.strip() for x in re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", t)]
    b = [x for x in b if x]  # the model sometimes echoes the instruction's empty box
    return b[-1] if b else None


def norm(x):
    return re.sub(r"\\?[$%]|\s|,|\\!|dfrac|tfrac|\\text\{[^}]*\}", "", x or "").replace("frac", "").rstrip(".").lower()


def ci(x, y, seed=0):
    d = [p - q for p, q in zip(x, y)]
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    return dict(diff=sum(d) / len(d), low=m[249], high=m[9749], better=sum(v > 0 for v in d), worse=sum(v < 0 for v in d))
