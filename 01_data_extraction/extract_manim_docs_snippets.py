#!/usr/bin/env python3
"""
Extract Manim-like code snippets from docs_html.jsonl and save as JSONL without using OpenAI.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Iterable, Tuple

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


def iter_code_blocks_from_html(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    for pre in soup.select('div.highlight pre'):
        text = pre.get_text()
        yield text, pre
    for code in soup.select('pre > code'):
        text = code.get_text()
        yield text, code
    for pre in soup.find_all('pre'):
        text = pre.get_text()
        yield text, pre


def extract_context_around(node, max_words: int = 200) -> Tuple[str, str]:
    texts_before = []
    texts_after = []
    for sib in node.find_all_previous(['p', 'li', 'dt', 'dd', 'h1', 'h2', 'h3'], limit=6):
        t = sib.get_text(" ", strip=True)
        if t:
            texts_before.append(t)
    for sib in node.find_all_next(['p', 'li', 'dt', 'dd', 'h1', 'h2', 'h3'], limit=6):
        t = sib.get_text(" ", strip=True)
        if t:
            texts_after.append(t)
    before_words = " ".join(reversed(texts_before)).split()
    after_words = " ".join(texts_after).split()
    return " ".join(before_words[:max_words]), " ".join(after_words[:max_words])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="docs_html.jsonl")
    p.add_argument("--out", default="manim_docs_snippets.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Max snippets to save; 0 = no limit")
    p.add_argument("--min_lines", type=int, default=3, help="Minimum code lines to consider a snippet")
    p.add_argument("--include_non_manim", action="store_true", help="If set, do not filter to manim-like code")
    args = p.parse_args()

    seen = set()
    kept = 0
    scanned = 0

    with open(args.out, 'w', encoding='utf-8') as fout:
        with open(args.input, 'r', encoding='utf-8') as fin:
            for line in fin:
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
                    if not args.include_non_manim and not is_manim_like(code):
                        continue
                    if len(code.splitlines()) < args.min_lines:
                        continue
                    h = hashlib.sha1(code.encode('utf-8')).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    before, after = extract_context_around(node, max_words=200)
                    rec = {
                        "url": url,
                        "depth": depth,
                        "code": code,
                        "before_context": before,
                        "after_context": after,
                        "code_sha1": h,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if args.limit and kept >= args.limit:
                        print(f"Limit reached ({args.limit}); stopping early.")
                        print(f"Saved {kept} snippets to {args.out} (code blocks scanned: {scanned})")
                        return

    print(f"Saved {kept} snippets to {args.out} (code blocks scanned: {scanned})")


if __name__ == '__main__':
    main()


