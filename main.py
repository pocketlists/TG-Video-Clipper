#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "work"}
_SAFE_TOP_NAMES = {"__name__", "__file__", "__doc__", "__package__", "__all__"}


def _sanitize_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return

    lines = raw.splitlines()
    changed = False

    # Strip any stray single-identifier line (e.g. 'p') before __future__ or at file edges
    cleaned_lines = []
    for line in lines:
        if re.match(r"^\s*[a-zA-Z_]\w*\s*$", line) and line.strip() not in _SAFE_TOP_NAMES:
            sys.stderr.write(f"auto-fixed stray token {line.strip()!r} in {path.relative_to(BASE_DIR)}\n")
            changed = True
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines) + ("\n" if raw.endswith("\n") else "")

    try:
        tree = ast.parse(text, filename=str(path))
        drop_lines: set[int] = set()
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
                if node.value.id not in _SAFE_TOP_NAMES:
                    for ln in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                        drop_lines.add(ln)
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Name):
                        for ln in range(sub.lineno, getattr(sub, "end_lineno", sub.lineno) + 1):
                            drop_lines.add(ln)

        if drop_lines:
            final_lines = [
                ln_text for idx, ln_text in enumerate(text.splitlines(), start=1)
                if idx not in drop_lines
            ]
            text = "\n".join(final_lines) + "\n"
            changed = True
    except SyntaxError:
        pass

    if changed:
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass


def _sanitize_project() -> None:
    for py_file in BASE_DIR.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        if py_file.resolve() == Path(__file__).resolve():
            continue
        _sanitize_file(py_file)


def _fail(message: str) -> int:
    sys.stderr.write(f"fatal: {message}\n")
    return 1


def main() -> int:
    _sanitize_project()
    try:
        from telegram.bot import main as bot_main
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _fail(f"cannot import telegram.bot ({type(exc).__name__}: {exc})")

    try:
        bot_main()
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _fail(f"telegram.bot.main() raised {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
