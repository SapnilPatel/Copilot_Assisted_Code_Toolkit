#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import difflib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


# ---------------------------
# LLM Interface (pluggable)
# ---------------------------

class LLM:
    """Minimal interface. Swap this with a real provider if desired."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLM):
    """
    Safe default: returns deterministic placeholder output so the project runs offline.
    Replace with your real LLM integration if you want.
    """
    def generate(self, prompt: str) -> str:
        if "UNIFIED DIFF" in prompt:
            # Placeholder diff: DOES NOT apply cleanly; it's here so workflow is demonstrable.
            return (
                "--- a/examples/legacy_module.py\n"
                "+++ b/examples/legacy_module.py\n"
                "@@ -1,3 +1,5 @@\n"
                "-# legacy_module.py\n"
                "+# legacy_module.py\n"
                "+# Refactor placeholder (replace this diff with an IDE-assisted refactor)\n"
                "+\n"
            )
        if "PYTEST FILE" in prompt:
            return (
                "import pytest\n"
                "from examples.legacy_module import process_data, average\n\n"
                "def test_process_data_basic():\n"
                "    assert process_data([1, 2, 3]) == [2, 4, 6]\n\n"
                "def test_average_basic():\n"
                "    assert average([2, 4, 6]) == 4\n"
            )
        if "MARKDOWN DOC" in prompt:
            return (
                "# legacy_module\n\n"
                "## Summary\n"
                "Placeholder documentation. Plug in a real LLM for richer docs.\n"
            )
        return "PLACEHOLDER OUTPUT"


def build_llm_from_env() -> LLM:
    """
    Uses environment variables to choose a provider.
    For now, default is MockLLM. Replace with your own implementation.

    Example idea (not implemented here):
      if os.getenv("LLM_PROVIDER") == "http":
          return HTTPChatLLM(base_url=os.getenv("LLM_URL"), api_key=os.getenv("LLM_KEY"))
    """
    return MockLLM()


# ---------------------------
# Retrieval (simple + fast)
# ---------------------------

@dataclasses.dataclass
class Snippet:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float


def iter_text_files(repo: Path, exts: Tuple[str, ...] = (".py", ".md", ".txt")) -> Iterable[Path]:
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"} for part in p.parts):
            continue
        yield p


def score_snippet(query_terms: List[str], text: str) -> float:
    t = text.lower()
    hits = sum(t.count(q) for q in query_terms)
    unique = sum(1 for q in set(query_terms) if q in t)
    return hits + 0.5 * unique


def retrieve(repo: Path, query: str, top_k: int = 6, window: int = 40) -> List[Snippet]:
    query_terms = [q.strip().lower() for q in re.split(r"\W+", query) if q.strip()]
    if not query_terms:
        return []

    results: List[Snippet] = []
    for fp in iter_text_files(repo):
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = raw.splitlines()
        idxs = []
        for i, line in enumerate(lines):
            ll = line.lower()
            if any(q in ll for q in query_terms):
                idxs.append(i)

        for i in idxs[:30]:
            start = max(0, i - window // 2)
            end = min(len(lines), i + window // 2)
            excerpt = "\n".join(lines[start:end])
            s = score_snippet(query_terms, excerpt)
            if s > 0:
                results.append(Snippet(
                    path=str(fp.relative_to(repo)),
                    start_line=start + 1,
                    end_line=end,
                    text=excerpt,
                    score=s
                ))

    results.sort(key=lambda x: x.score, reverse=True)

    seen = set()
    uniq: List[Snippet] = []
    for r in results:
        key = (r.path, r.start_line)
        if key not in seen:
            uniq.append(r)
            seen.add(key)
        if len(uniq) >= top_k:
            break
    return uniq


def format_snippets(snips: List[Snippet]) -> str:
    blocks = []
    for s in snips:
        blocks.append(
            f"[{s.path}:{s.start_line}-{s.end_line} | score={s.score:.1f}]\n"
            f"{s.text}\n"
        )
    return "\n".join(blocks).strip()


# ---------------------------
# File helpers
# ---------------------------

def read_file(repo: Path, rel: str) -> str:
    p = repo / rel
    return p.read_text(encoding="utf-8", errors="ignore")


def write_file(repo: Path, rel: str, content: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# ---------------------------
# Docs template (fallback)
# ---------------------------

DEFAULT_DOC_TEMPLATE = """\
# {module_name}

## Summary
{summary}

## Key APIs / Entry Points
{apis}

## How it works
{how_it_works}

## Usage Examples
{examples}

## Notes / Assumptions
{notes}
"""


# ---------------------------
# Prompts
# ---------------------------

def prompt_refactor(target_rel: str, target_code: str, context: str) -> str:
    return textwrap.dedent(f"""\
    You are a senior software engineer.

    TASK:
    Produce a UNIFIED DIFF to refactor the file below. Goals:
    - Improve readability, structure, and maintainability
    - Keep behavior identical unless clearly unsafe/buggy
    - Add/adjust docstrings and type hints where valuable
    - Avoid large rewrites; prefer incremental improvements

    IMPORTANT OUTPUT RULES:
    - Output ONLY a valid UNIFIED DIFF
    - Use paths exactly as: --- a/{target_rel} and +++ b/{target_rel}
    - Do not include commentary outside the diff

    REPO CONTEXT SNIPPETS:
    {context}

    TARGET FILE: {target_rel}
    ```python
    {target_code}
    ```
    """)


def prompt_tests(module_rel: str, module_code: str, context: str) -> str:
    return textwrap.dedent(f"""\
    You are a test engineer.

    TASK:
    Generate a PYTEST FILE for the module below.
    Requirements:
    - Use pytest
    - Include edge cases and at least one negative test when applicable
    - Prefer small, deterministic tests
    - If external dependencies exist, mock them
    - Keep it minimal but meaningful

    IMPORTANT OUTPUT RULES:
    - Output ONLY the Python code for the test file (no markdown fences)

    REPO CONTEXT SNIPPETS:
    {context}

    MODULE: {module_rel}
    ```python
    {module_code}
    ```
    """)


def prompt_docs(module_rel: str, module_code: str, context: str) -> str:
    return textwrap.dedent(f"""\
    You are a technical writer for an engineering team.

    TASK:
    Produce a MARKDOWN DOC for this module suitable for internal use.
    Include:
    - Summary
    - Key functions/classes and what they do
    - Expected inputs/outputs
    - Usage examples (short)
    - Notes/assumptions

    IMPORTANT OUTPUT RULES:
    - Output ONLY markdown (no backticks around the entire doc)

    REPO CONTEXT SNIPPETS:
    {context}

    MODULE: {module_rel}
    ```python
    {module_code}
    ```
    """)


# ---------------------------
# Core actions
# ---------------------------

def do_refactor(repo: Path, target: str, out_patch: str, llm: LLM) -> None:
    target_rel = str(Path(target))
    code = read_file(repo, target_rel)
    ctx = format_snippets(retrieve(repo, query=f"{target_rel} functions classes usage", top_k=6))
    p = prompt_refactor(target_rel, code, ctx)

    diff_text = llm.generate(p).strip() + "\n"
    outp = Path(out_patch)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(diff_text, encoding="utf-8")
    print(f"[OK] Patch written to {out_patch}")
    print("[NOTE] Use Copilot/Cursor/Windsurf in your IDE to refine this patch into a real refactor.")


def do_gen_tests(repo: Path, module: str, tests_out: str, llm: LLM) -> None:
    module_rel = str(Path(module))
    code = read_file(repo, module_rel)
    ctx = format_snippets(retrieve(repo, query=f"{module_rel} usage examples tests", top_k=6))
    p = prompt_tests(module_rel, code, ctx)

    test_code = llm.generate(p).rstrip() + "\n"
    write_file(repo, tests_out, test_code)
    print(f"[OK] Tests written to {tests_out}")
    print("[NOTE] Use Copilot/Cursor/Windsurf to improve coverage and edge cases.")


def do_gen_docs(repo: Path, module: str, out_md: str, llm: LLM) -> None:
    module_rel = str(Path(module))
    code = read_file(repo, module_rel)
    ctx = format_snippets(retrieve(repo, query=f"{module_rel} how to use API", top_k=6))
    p = prompt_docs(module_rel, code, ctx)

    md = llm.generate(p).rstrip() + "\n"
    if len(md.splitlines()) < 10:
        module_name = Path(module_rel).stem
        md = DEFAULT_DOC_TEMPLATE.format(
            module_name=module_name,
            summary="Short summary of the module purpose.",
            apis="- process_data(data)\n- average(values)",
            how_it_works="High-level explanation of the flow and decisions.",
            examples="```python\nfrom examples.legacy_module import process_data, average\n\nx = process_data([1, None, 3])\nprint(average(x))\n```",
            notes="This module is intentionally simple and used as a demo target.",
        )

    write_file(repo, out_md, md)
    print(f"[OK] Docs written to {out_md}")


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)


def do_qa(repo: Path, run_pytest: bool = True, pytest_args: Optional[List[str]] = None) -> None:
    rc = 0

    # Syntax check for .py
    py_files = [str(p.relative_to(repo)) for p in iter_text_files(repo, exts=(".py",))]
    for f in py_files[:300]:
        content = read_file(repo, f)
        try:
            compile(content, f, "exec")
        except SyntaxError as e:
            print(f"[FAIL] Syntax error in {f}: {e}")
            rc = 1

    if run_pytest:
        args = ["pytest"]
        if pytest_args:
            args += pytest_args
        prc = run_cmd(args, repo)
        rc = rc or prc

    if rc == 0:
        print("[OK] QA passed")
    else:
        print("[WARN] QA had issues (see logs)")
        sys.exit(rc)


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Copilot-Assisted Code Toolkit (MVP)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ref = sub.add_parser("refactor", help="Generate a unified diff patch for a target file")
    p_ref.add_argument("--repo", required=True, help="Repo root")
    p_ref.add_argument("--target", required=True, help="Target file path (relative to repo)")
    p_ref.add_argument("--out", required=True, help="Output patch file path")

    p_tests = sub.add_parser("gen-tests", help="Generate pytest tests for a module")
    p_tests.add_argument("--repo", required=True, help="Repo root")
    p_tests.add_argument("--module", required=True, help="Module file path (relative to repo)")
    p_tests.add_argument("--tests", required=True, help="Output tests file path (relative to repo), e.g., tests/test_x.py")

    p_docs = sub.add_parser("gen-docs", help="Generate markdown docs for a module")
    p_docs.add_argument("--repo", required=True, help="Repo root")
    p_docs.add_argument("--module", required=True, help="Module file path (relative to repo)")
    p_docs.add_argument("--out", required=True, help="Output markdown path (relative to repo), e.g., docs/x.md")

    p_qa = sub.add_parser("qa", help="Run basic QA checks (syntax + pytest optional)")
    p_qa.add_argument("--repo", required=True, help="Repo root")
    p_qa.add_argument("--pytest", action="store_true", help="Run pytest")
    p_qa.add_argument("--pytest-args", nargs="*", default=[], help="Extra args for pytest")

    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    llm = build_llm_from_env()

    if args.cmd == "refactor":
        do_refactor(repo, args.target, args.out, llm)
    elif args.cmd == "gen-tests":
        do_gen_tests(repo, args.module, args.tests, llm)
    elif args.cmd == "gen-docs":
        do_gen_docs(repo, args.module, args.out, llm)
    elif args.cmd == "qa":
        do_qa(repo, run_pytest=bool(args.pytest), pytest_args=args.pytest_args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
