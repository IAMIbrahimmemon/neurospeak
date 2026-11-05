"""NeuroSpeak: read and write a transformer's neurons as words.

    import neurospeak as ns
    brain = ns.load()
    rep = brain.concept("repetition", [ns.pair(is_="…", not_="…", ask="…")], at=[15, 19])
    print(rep.where())
"""
from .anatomy import Anatomy, discover
from .core import (DEFAULT_MODEL, MATH_SUFFIX, Addr, Brain, Branch, Concept, Edit, Expectation, ExpectationFailed, Head, Neuron, Pair,
                   apply_edit_file, load, load_edit_file, loopiness, pair, parse_addr)
from .lang import Interpreter

__all__ = ["Anatomy", "DEFAULT_MODEL", "MATH_SUFFIX", "discover", "Addr", "Brain", "Branch", "Concept", "Edit", "Expectation", "ExpectationFailed", "Head", "Interpreter",
           "Neuron", "Pair", "apply_edit_file", "load", "load_edit_file", "loopiness", "pair", "parse_addr"]
