#!/usr/bin/env python3
"""
Build a small instruction to code sample from docs_html.jsonl (HTML pages).

Approach:
- Parse each JSONL line: {url, depth, content=HTML}
- Use BeautifulSoup to locate true Python code blocks (<pre><code ...>, Sphinx highlight)
- Optionally call an LLM (when --api_key_file is provided) with ±200-word HTML context around each code block
  to synthesize one concise, precise instruction describing the visual outcome
- On LLM failure or when no API key is given, fall back to a deterministic (regex-based) instruction generator
- Save up to N instruction to code pairs to JSONL
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Iterable, Tuple, List

from bs4 import BeautifulSoup


def normalize_code(code: str) -> str:
    code = re.sub(r"^\s*>>> ?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*\.\.\. ?", "", code, flags=re.MULTILINE)
    lines = [ln.rstrip() for ln in code.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def is_manim_like(code: str) -> bool:
    needles = [
        r"\bfrom\s+manim\s+import\b", r"\bimport\s+manim\b",
        r"\bclass\s+\w+\(.*?Scene\)\s*:",
        r"\bCircle\(|\bSquare\(|\bTriangle\(|\bMathTex\(|\bText\("
    ]
    return any(re.search(p, code) for p in needles)


def iter_code_blocks_from_html(html: str) -> Iterable[tuple[str, object]]:
    soup = BeautifulSoup(html, 'html.parser')
    # Common Sphinx patterns:
    # - <div class="highlight-python notranslate"><div class="highlight"><pre>...</pre></div></div>
    # - <pre><code class="language-python">...</code></pre>
    # - <pre>...</pre>
    # Collect candidate pre/code blocks broadly, then filter later.

    # 1) div.highlight pre
    for pre in soup.select('div.highlight pre'):
        text = pre.get_text()
        yield text, pre

    # 2) pre > code
    for code in soup.select('pre > code'):
        text = code.get_text()
        yield text, code

    # 3) any bare <pre> blocks
    for pre in soup.find_all('pre'):
        text = pre.get_text()
        yield text, pre


def extract_context_around(node, max_words: int = 200) -> tuple[str, str]:
    # Collect a few nearby blocks of text before/after the code block
    texts_before: list[str] = []
    texts_after: list[str] = []
    # previous siblings/sections
    for sib in node.find_all_previous(['p', 'li', 'dt', 'dd', 'h1', 'h2', 'h3'], limit=6):
        t = sib.get_text(" ", strip=True)
        if t:
            texts_before.append(t)
    # next siblings/sections
    for sib in node.find_all_next(['p', 'li', 'dt', 'dd', 'h1', 'h2', 'h3'], limit=6):
        t = sib.get_text(" ", strip=True)
        if t:
            texts_after.append(t)
    before_words = " ".join(reversed(texts_before)).split()
    after_words = " ".join(texts_after).split()
    return " ".join(before_words[:max_words]), " ".join(after_words[:max_words])


def instruction_from_llm(client, code: str, before: str, after: str) -> str:
    system = (
        "You are a Manim expert. Return ONE concise, precise instruction (≤60 words) that describes the visual result of the code: objects, properties (colors/sizes/positions), animations, timing, camera ops. No code."
    )
    user = (
        f"Context before:\n{before}\n\nCode:\n```python\n{code}\n```\n\nContext after:\n{after}\n"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1, max_tokens=160,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()


def generate_instruction(code: str) -> str:
    parts: List[str] = []
    # Objects
    m = re.search(r"Circle\(.*?radius\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code)
    if m:
        parts.append(f"Create a circle with radius {m.group(1)} units and {m.group(2).lower()} color")
    elif re.search(r"Circle\(", code):
        parts.append("Create a circle")

    m = re.search(r"Square\(.*?side_length\s*=\s*([\d\.]+).*?color\s*=\s*(\w+)", code)
    if m:
        parts.append(f"Create a square with side length {m.group(1)} units and {m.group(2).lower()} color")
    elif re.search(r"Square\(", code):
        parts.append("Create a square")

    m = re.search(r"Text\(\s*['\"]([^'\"]+)['\"]", code)
    if m:
        parts.append(f"Create a text displaying '{m.group(1)}'")

    if re.search(r"Axes\(", code):
        parts.append("Create axes")

    # Animations and timing
    m = re.search(r"\.animate\.shift\(RIGHT\s*\*\s*([\d\.]+)\)", code)
    if m:
        parts.append(f"move object {m.group(1)} units to the right")

    m = re.search(r"\.animate\.scale\(([\d\.]+)\)", code)
    if m:
        parts.append(f"scale object by factor {m.group(1)}")

    # Degrees: handle 90 * DEGREES and -90 * DEGREES
    m = re.search(r"\.animate\.rotate\(([-\d\.]+)\s*\*\s*DEGREES\)", code)
    if m:
        parts.append(f"rotate object by {abs(float(m.group(1))):g} degrees" + (" counterclockwise" if float(m.group(1)) < 0 else ""))

    # radians= form
    m = re.search(r"Rotating\([^)]*radians\s*=\s*([\d\.]+)\b", code)
    if m:
        parts.append(f"rotate object by {m.group(1)} radians")

    m = re.search(r"run_time\s*=\s*([\d\.]+)", code)
    if m and parts:
        parts[-1] = parts[-1] + f" over {m.group(1)} seconds"

    if not parts:
        return "Create a basic Manim scene."
    out = ". ".join(parts)
    if not out.endswith('.'):
        out += '.'
    return out[0].upper() + out[1:]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="docs_html.jsonl")
    p.add_argument("--out", default="manim_docs_html_pairs.sample.jsonl")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--api_key_file", default=None)
    args = p.parse_args()

    seen = set()
    kept = 0
    scanned = 0

    client = None
    if args.api_key_file:
        try:
            import openai
            api_key = Path(args.api_key_file).read_text(encoding='utf-8').strip()
            client = openai.OpenAI(api_key=api_key)
        except Exception:
            client = None

    with open(args.out, 'w', encoding='utf-8') as fout:
        with open(args.input, 'r', encoding='utf-8') as fin:
            for line in fin:
                if kept >= args.limit:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                url = obj.get('url')
                depth = obj.get('depth')
                html = obj.get('content', '')

                for raw, node in iter_code_blocks_from_html(html):
                    scanned += 1
                    code = normalize_code(raw)
                    if not is_manim_like(code):
                        continue
                    if len(code.splitlines()) < 3:
                        continue
                    h = hashlib.sha1(code.encode('utf-8')).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    instr: str
                    if client is not None:
                        try:
                            before, after = extract_context_around(node, max_words=200)
                            instr = instruction_from_llm(client, code, before, after)
                        except Exception:
                            instr = generate_instruction(code)
                    else:
                        instr = generate_instruction(code)
                    rec = {"url": url, "depth": depth, "instruction": instr, "code": code}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= args.limit:
                        break

    print(f"Saved {kept} pairs to {args.out} (code blocks scanned: {scanned})")


if __name__ == '__main__':
    main()


