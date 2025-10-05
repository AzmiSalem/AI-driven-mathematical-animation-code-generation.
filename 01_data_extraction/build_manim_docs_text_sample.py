#!/usr/bin/env python3
"""
Build a small sample instruction→code dataset from docs_text.jsonl (plain text pages).

Strategy (no LLM):
- Read each JSONL line: {url, depth, content}
- Extract code snippets via regex:
  - doctest blocks: contiguous ">>>" / "..." lines
  - python script blocks inferred by presence of "from manim" / "import manim"
    and a following class Scene or typical Manim API usage
- Clean snippets (strip prompts, trim whitespace)
- Generate a concise, specific instruction using deterministic heuristics
- Save up to N results to a JSONL file
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Iterable, Tuple, Dict, List


def normalize_code(code: str) -> str:
    # Strip doctest prompts
    code = re.sub(r"^\s*>>> ?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*\.\.\. ?", "", code, flags=re.MULTILINE)
    # Drop trailing outputs or REPL echo lines that don't look like code (best-effort)
    lines = [ln.rstrip() for ln in code.splitlines()]
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def is_manim_like(code: str) -> bool:
    patterns = [
        r"\bfrom\s+manim\s+import\b",
        r"\bimport\s+manim\b",
        r"\bclass\s+\w+\(.*?Scene\)\s*:\b",
        r"\bCircle\(|\bSquare\(|\bTriangle\(|\bMathTex\(|\bText\("
    ]
    return any(re.search(p, code) for p in patterns)


def iter_doctest_blocks(text: str) -> Iterable[str]:
    # Capture blocks starting with ">>>" and continued by "..."
    pattern = re.compile(r"(>>> .*(?:\n(?:>>>|\.\.\.).*)*)", re.MULTILINE)
    for m in pattern.finditer(text):
        yield m.group(1)


def iter_script_blocks(text: str) -> Iterable[str]:
    # Heuristic: grab segments that include "from manim" or "import manim" and
    # extend to the next double-blank or section marker.
    # Also capture typical Manim class Scene blocks.
    candidates: List[Tuple[int, int]] = []

    for m in re.finditer(r"from\s+manim\s+import\s+\*|import\s+manim", text):
        start = max(0, text.rfind("\n", 0, m.start()))
        # end at the next two consecutive newlines or end of text
        dbl = text.find("\n\n", m.end())
        end = dbl if dbl != -1 else len(text)
        candidates.append((start, end))

    for m in re.finditer(r"\bclass\s+\w+\(.*?Scene\)\s*:\s*", text):
        start = max(0, text.rfind("\n", 0, m.start()))
        dbl = text.find("\n\n", m.end())
        end = dbl if dbl != -1 else len(text)
        candidates.append((start, end))

    # Merge overlapping ranges
    candidates.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in candidates:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][1] and merged[-1][0] or s, max(merged[-1][1], e))

    for s, e in merged:
        snippet = text[s:e]
        yield snippet


def generate_instruction(code: str) -> str:
    parts = []
    # Objects
    if re.search(r"Circle\(.*?radius\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code):
        m = re.search(r"Circle\(.*?radius\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code)
        radius, color = m.group(1), m.group(2)
        parts.append(f"Create a circle with radius {radius} units and {color.lower()} color")
    elif re.search(r"Circle\(", code):
        parts.append("Create a circle")

    if re.search(r"Square\(.*?side_length\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code):
        m = re.search(r"Square\(.*?side_length\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code)
        sl, color = m.group(1), m.group(2)
        parts.append(f"Create a square with side length {sl} units and {color.lower()} color")
    elif re.search(r"Square\(", code):
        parts.append("Create a square")

    if re.search(r"Text\(\s*['\"]([^'\"]+)['\"]", code):
        txt = re.search(r"Text\(\s*['\"]([^'\"]+)['\"]", code).group(1)
        parts.append(f"Create a text displaying '{txt}'")

    if re.search(r"Axes\(", code):
        parts.append("Create axes")

    # Animations
    if re.search(r"\.animate\.shift\(RIGHT\s*\*\s*([\d\.]+)\)", code):
        d = re.search(r"\.animate\.shift\(RIGHT\s*\*\s*([\d\.]+)\)", code).group(1)
        parts.append(f"move object {d} units to the right")
    if re.search(r"\.animate\.scale\(([\d\.]+)\)", code):
        f = re.search(r"\.animate\.scale\(([\d\.]+)\)", code).group(1)
        parts.append(f"scale object by factor {f}")
    if re.search(r"\.animate\.rotate\(([-\d\.]+)\s*\*\s*DEGREES\)", code):
        deg = re.search(r"\.animate\.rotate\(([-\d\.]+)\s*\*\s*DEGREES\)", code).group(1)
        parts.append(f"rotate object by {deg} degrees")

    if re.search(r"run_time\s*=\s*([\d\.]+)", code):
        t = re.search(r"run_time\s*=\s*([\d\.]+)", code).group(1)
        # append timing to last animation if exists
        if parts:
            parts[-1] = parts[-1] + f" over {t} seconds"

    if not parts:
        return "Create a basic Manim scene"
    # Capitalize first and end with period. Join with ". "
    out = ". ".join(parts)
    if not out.endswith("."):
        out += "."
    return out[0].upper() + out[1:]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="docs_text.jsonl")
    p.add_argument("--out", default="manim_docs_text_pairs.sample.jsonl")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    seen_hashes = set()
    kept = 0
    total_candidates = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        with open(args.input, "r", encoding="utf-8") as fin:
            for line in fin:
                if kept >= args.limit:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                url = obj.get("url")
                depth = obj.get("depth")
                content = obj.get("content", "")

                # Extract doctest blocks
                for block in iter_doctest_blocks(content):
                    total_candidates += 1
                    code = normalize_code(block)
                    if not is_manim_like(code):
                        continue
                    if len(code.splitlines()) < 3:
                        continue
                    h = hashlib.sha1(code.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    instr = generate_instruction(code)
                    rec = {"url": url, "depth": depth, "instruction": instr, "code": code}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= args.limit:
                        break

                if kept >= args.limit:
                    break

                # Extract script-style blocks
                for block in iter_script_blocks(content):
                    total_candidates += 1
                    code = normalize_code(block)
                    if not is_manim_like(code):
                        continue
                    if len(code.splitlines()) < 3:
                        continue
                    h = hashlib.sha1(code.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    instr = generate_instruction(code)
                    rec = {"url": url, "depth": depth, "instruction": instr, "code": code}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= args.limit:
                        break

    print(f"Saved {kept} pairs to {args.out} (candidates scanned: {total_candidates})")


if __name__ == "__main__":
    main()


