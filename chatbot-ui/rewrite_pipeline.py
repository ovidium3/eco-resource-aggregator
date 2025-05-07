#!/usr/bin/env python
"""Command‑line entry‑point: classify and rewrite in one shot.

Example:
    python rewrite_pipeline.py "impact of rising sea levels on coastal farming"
"""
import argparse, textwrap
from zero_shot_classifier import predict_category
from transformer_rewriter import rewrite_query

def main():
    p = argparse.ArgumentParser(
        description="Zero‑shot classify a climate query, then rewrite it."
    )
    p.add_argument("query", nargs="+", help="raw query string")
    args = p.parse_args()

    raw = " ".join(args.query)
    cat = predict_category(raw)
    rewritten = rewrite_query(raw, cat)

    # NEW: show class with underscores
    cat_print = cat.replace(" ", "_")

    print(textwrap.dedent(f"""
        Raw query       : {raw}
        Predicted class : {cat_print}
        Rewritten query : {rewritten}
    """).strip())

def doPipeline(query):
    cat = predict_category(query)
    rewritten = rewrite_query(query, cat)
    cat_print = cat.replace(" ", "_")
    return cat_print, rewritten, query

if __name__ == "__main__":
    main()