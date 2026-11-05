"""Value-based answer check used by the sensitivity and audit scripts."""
import re
from gsm_score import boxed, norm


def value(x):
    m = re.fullmatch(r"-?\d+(\.\d+)?", norm(x) or "")
    return float(m.group()) if m else None


def right(r):
    b, a = boxed(r["completion"]), r["answer"]
    if b is None:
        return False
    return norm(b) == norm(a) or (value(b) is not None and value(b) == value(a))
