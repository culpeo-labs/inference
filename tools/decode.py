#!/usr/bin/env python3
"""Detokenize generated token ids back into text.

Reads whitespace- or comma-separated token ids (from argv); or one per line from stdin.
Prints the detokenized text to stdout.
"""
import sys
import re

from transformers import AutoTokenizer


def main():
    ckpt = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(ckpt)

    if len(sys.argv) > 2:
        raw = " ".join(sys.argv[2:])
        ids = [int(t) for t in re.split(r"[\s,]+", raw.strip()) if t]
        if not ids:
            print("(no ids given)", file=sys.stderr)
            return
        text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(text)
    else:
        for line in sys.stdin:
            n = int(line)
            text = tok.decode(n, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(text, end="", )
        print()


if __name__ == "__main__":
    main()