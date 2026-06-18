#!/usr/bin/env python3
import ast
import sys

MAX_DOC_LINES = 6


def violations(path: str, source: str) -> list[str]:
    """Return style violations: module docstrings and over-long docstrings."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    out: list[str] = []
    if ast.get_docstring(tree) is not None:
        out.append(
            f"{path}: module-level docstring not allowed (move it into a class/function or drop it)"
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        n = len(doc.splitlines())
        if n > MAX_DOC_LINES:
            out.append(
                f"{path}:{node.lineno}: docstring for '{node.name}' is {n} lines (max {MAX_DOC_LINES})"
            )
    return out


def demo() -> None:
    assert violations("x.py", '"""mod."""\n') == [
        "x.py: module-level docstring not allowed (move it into a class/function or drop it)"
    ]
    assert violations("x.py", "def f():\n    pass\n") == []
    long = '"""' + "\n".join(str(i) for i in range(MAX_DOC_LINES + 2)) + '"""'
    assert violations("x.py", f"def f():\n    {long}\n    pass\n")
    assert violations("x.py", 'def f():\n    """one line."""\n    pass\n') == []
    print("ok")


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        demo()
        return 0
    found: list[str] = []
    for path in argv:
        with open(path, encoding="utf-8") as fh:
            found += violations(path, fh.read())
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
