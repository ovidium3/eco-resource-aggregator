#!/usr/bin/env python3
import json
import numpy as np
import re
from itertools import islice

from datasets import load_dataset
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def extract_terms_from_raw_text(dataset):
    terms = set()
    for example in tqdm(dataset, desc="Extracting NER terms"):
        for line in example.get("text", "").split("\n"):
            parts = line.strip().split()
            if len(parts) == 2:
                token, tag = parts
                if tag != "O" and re.match(r"^[A-Za-z]{3,}$", token):
                    terms.add(token.lower())
    return list(terms)


def main():
    # --- AUTHENTICATE TO HF (for gated datasets) ---
    login(token="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")

    # --- SETTINGS ---
    SAMPLE_SIZE    = 1000                # how many docs to scan in this run
    SIM_THRESHOLD  = 0.25                # cosine‐sim cutoff for categories
    OUTPUT_FILE    = "classified_papers.jsonl"
    MODEL_NAME     = "climatebert/distilroberta-base-climate-f"

    # --- STEP 1: Extract keywords from Climate-Change-NER (optional) ---
    print("Loading Climate-Change-NER keywords...")
    ner_dataset = load_dataset("ibm-research/Climate-Change-NER", split="train")
    CLIMATE_KEYWORDS = extract_terms_from_raw_text(ner_dataset)
    print(f"Extracted {len(CLIMATE_KEYWORDS)} climate NER tokens.")

    # --- STEP 2: Define your 13 climate categories ---
    CATEGORIES = [
        "climate-assets",
        "climate-datasets",
        "climate-greenhouse-gases",
        "climate-hazards",
        "climate-impacts",
        "climate-mitigations",
        "climate-models",
        "climate-nature",
        "climate-observations",
        "climate-organisms",
        "climate-organizations",
        "climate-problem-origins",
        "climate-properties",
    ]
    print(f"Using {len(CATEGORIES)} predefined categories.")

    # --- STEP 3: Load ClimateBERT to embed categories + docs ---
    print("Loading ClimateBERT model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding categories...")
    cat_embs = model.encode(
        CATEGORIES,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    # shape: (13, 768)

    # --- STEP 4: Stream papers and classify on the fly ---
    print(f"Streaming up to {SAMPLE_SIZE} papers …")
    stream = islice(
        load_dataset("allenai/pes2o", split="train", streaming=True),
        SAMPLE_SIZE,
    )

    out_f = open(OUTPUT_FILE, "w")
    kept = 0

    for paper in tqdm(stream, total=SAMPLE_SIZE, desc="Classifying"):
        # build text blob
        text = " ".join(paper.get(f, "") for f in ("title", "abstract", "text"))
        if not text.strip():
            continue

        # embed document
        doc_emb = model.encode(text, normalize_embeddings=True)

        # cosine similarity = dot product (embeddings are normalized)
        sims = np.dot(cat_embs, doc_emb)  # shape: (13,)

        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        # only keep if above threshold
        if best_sim >= SIM_THRESHOLD:
            paper["climate_category"] = CATEGORIES[best_idx]
            paper["category_score"]   = best_sim
            out_f.write(json.dumps(paper) + "\n")
            kept += 1

    out_f.close()
    print(f"\nDone. Classified {kept} papers (out of {SAMPLE_SIZE}) saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
