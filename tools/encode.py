#!/usr/bin/env python3
"""Tokenize text into token ids.

Reads from argv and outputs token ids to stdout.
"""
import sys
import re

from transformers import AutoTokenizer


def main():
    ckpt = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(ckpt)

    if len(sys.argv) > 2:
        raw = " ".join(sys.argv[2:])
        tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
        if not tokens:
            print("(no text given)", file=sys.stderr)
            return
        ids = tok.encode(" ".join(tokens))
        print(" ".join(map(str, ids)))
    else:
        print("(no text given)", file=sys.stderr)
        return


if __name__ == "__main__":
    main()