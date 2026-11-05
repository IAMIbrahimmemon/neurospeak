"""No model needed: the parts of the language a person can get wrong by typing."""
import pytest

import neurospeak as ns
from neurospeak.lang import Interpreter, lex
from neurospeak.reports import Measurement, Promise


def test_addresses_round_trip():
    for s in ("L19", "L19.mlp", "L15.attn", "L19.n11", "L15.h7"):
        assert str(ns.parse_addr(s)) == s


def test_bad_address_names_the_fix():
    with pytest.raises(SyntaxError, match="try L19"):
        ns.parse_addr("layer19")


def test_strings_span_lines_and_hide_comment_marks():
    lines, strs = lex('read x on """a # b\nc"""  # real comment\n')
    assert lines == [(0, "read x on §0§")]
    assert strs == ["a # b\nc"]


@pytest.mark.parametrize("delta,verb,kept", [(-0.05, "falls", True), (0.0, "falls", False), (0.05, "rises", True),
                                             (0.01, "steady", True), (0.05, "steady", False)])
def test_promise_verdicts(delta, verb, kept):
    m = Measurement("c", {"is": -1.0, "not": -1.0}, {"is": -1.0 + delta, "not": -1.0})
    assert bool(Promise("c", "is", verb, 0.02, m)) is kept


def test_loopiness_separates_loops_from_prose():
    assert ns.loopiness("wait let me check that again " * 40) > 0.9
    assert ns.loopiness(" ".join(f"word{i}" for i in range(200))) == 0


def test_malformed_statement_reports_usage_and_continues(capsys):
    it = Interpreter()
    assert it.run_source("frobnicate L19\n") == 0
    assert "unknown statement" in capsys.readouterr().out
