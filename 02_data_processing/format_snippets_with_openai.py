#!/usr/bin/env python3
"""
Read snippets JSONL (from extract_manim_docs_snippets.py) and produce instruction to code pairs using OpenAI.

"""

import json
from pathlib import Path
from typing import Optional


def get_client(api_key_file: str):
    import openai
    api_key = Path(api_key_file).read_text(encoding='utf-8').strip()
    return openai.OpenAI(api_key=api_key)


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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="manim_docs_snippets.jsonl")
    p.add_argument("--out", default="manim_docs_html_pairs.sample.jsonl")
    p.add_argument("--api_key_file", required=True)
    p.add_argument("--limit", type=int, default=0, help="Max pairs to produce; 0 = no limit")
    args = p.parse_args()

    client = get_client(args.api_key_file)

    written = 0
    scanned = 0
    with open(args.out, 'w', encoding='utf-8') as fout:
        with open(args.input, 'r', encoding='utf-8') as fin:
            for line in fin:
                scanned += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                code = obj.get('code', '')
                before = obj.get('before_context', '')
                after = obj.get('after_context', '')
                url = obj.get('url')
                depth = obj.get('depth')
                code_sha1 = obj.get('code_sha1')
                try:
                    instr = instruction_from_llm(client, code, before, after)
                except Exception as e:
                    # Skip on failure to ensure cost control and clean outputs
                    continue
                rec = {"url": url, "depth": depth, "instruction": instr, "code": code, "code_sha1": code_sha1}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    break

    print(f"Saved {written} pairs to {args.out} (snippets scanned: {scanned})")


if __name__ == '__main__':
    main()


