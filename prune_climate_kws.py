import json
import numpy as np
import os
import re

from datasets import load_dataset
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def extract_terms_from_raw_text(dataset):
    """
    Extracts terms from the raw text of the dataset.
    Args:
        dataset: The dataset to extract terms from.
    Returns:
        A list of unique terms extracted from the dataset.
    """
    terms = set()
    for example in tqdm(dataset, desc="Extracting NER terms"):
        for line in example.get("text", "").split("\n"):
            parts = line.strip().split()
            if len(parts) == 2:
                token, tag = parts
                if tag != "O" and re.match(r"^[A-Za-z]{3,}$", token):   # filter out non-relevant tag
                    terms.add(token.lower())
    return list(terms)


def prune_by_similarity(raw_keywords, anchors, model_name="climatebert/distilroberta-base-climate-f"):
    """
    Prune keywords by computing their similarity to anchor terms.
    Args:
        raw_keywords: The list of raw keywords to prune.
        anchors: The list of anchor terms to compare against.
        model_name: The name of the SentenceTransformer model to use.
    Returns:
        A list of pruned keywords that are similar to the anchor terms.
    """
    # 1) embed
    model    = SentenceTransformer(model_name)
    print("Embedding anchors…")
    anc_embs = model.encode(anchors, normalize_embeddings=True)
    print("Embedding keywords…")
    kw_embs  = model.encode(raw_keywords, normalize_embeddings=True)

    # 2) compute max‐similarity to any anchor for each keyword
    #    sims.shape = (n_keywords, n_anchors)
    sims = np.dot(kw_embs, anc_embs.T)
    max_sims = sims.max(axis=1)

    # 3) show some percentiles to pick a threshold
    for p in [50, 70, 80, 90, 95, 99]:
        print(f"{p}th percentile of max‑sim = {np.percentile(max_sims, p):.3f}")

    # 4) switch to centroid‑based sim
    centroid = anc_embs.mean(axis=0)
    cent_sims = kw_embs.dot(centroid)
    print(f"Centroid 50/90 percentiles: "
          f"{np.percentile(cent_sims,50):.3f}/"
          f"{np.percentile(cent_sims,90):.3f}")

    # 5) prune with chosen cutoff
    #    keeps 176/1742 --> roughly top 10% of keywords
    threshold = 0.977795
    keep = [kw for kw, ms in zip(raw_keywords, cent_sims) if ms >= threshold]
    print(f"Pruned down to {len(keep)} / {len(raw_keywords)} keywords")
    return keep


def main():
    """Extracts keywords from dataset. Prunes bottom 90% of keywords."""

    # AUTHENTICATE TO HUGGINGFACE (for gated datasets)
    login(token="YOUR_HUGGINGFACE_TOKEN")  # replace with your token

    # SETTINGS
    OUTPUT_FILE = "climate_papers.json"

    # extract keywords from Climate-Change-NER dataset
    print("Loading Climate-Change-NER keywords...")
    ner_dataset = load_dataset("ibm-research/Climate-Change-NER", split="train")
    raw_keywords = extract_terms_from_raw_text(ner_dataset)
    print(f"Extracted {len(raw_keywords)} keywords.")

    # 13 predefined categories from Climate-Change-NER
    anchors = [
        "climate assets",
        "climate datasets",
        "greenhouse gases",
        "climate hazards",
        "climate impacts",
        "climate mitigation",
        "climate models",
        "climate nature",
        "climate observations",
        "climate organisms",
        "climate organizations",
        "origins of climate problems",
        "climate properties",
    ]

    # print pruned keywords to output
    climate_words = prune_by_similarity(raw_keywords, anchors)
    with open("climate_kw.txt", "w") as f:
        json.dump(climate_words, f)

    return 0


if __name__ == "__main__":
    main()
