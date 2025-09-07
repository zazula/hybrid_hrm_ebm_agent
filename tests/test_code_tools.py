from pathlib import Path

from hybrid_agent.tools.code_tools import syntax_ok, patch_applies


def test_syntax_ok_valid_and_invalid():
    ok, err = syntax_ok("x=1\nprint(x)")
    assert ok and err is None

    ok, err = syntax_ok("def oops(:\n    pass")
    assert not ok and isinstance(err, str) and "SyntaxError" in err


def test_patch_applies_writes_file(tmp_path: Path):
    repo = tmp_path
    rel = "tmp_module.py"
    code = "def add(a,b):\n    return a+b\n"
    ok, err = patch_applies(str(repo), rel, code)
    assert ok and err is None
    p = repo / rel
    assert p.exists() and p.read_text(encoding="utf-8") == code
