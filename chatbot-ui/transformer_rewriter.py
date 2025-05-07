"""
transformer_rewriter.py  (v12 – better climate queries)

BART paraphraser for climate‑rich inputs; template for others.
"""

import re, collections, random
from functools import lru_cache
from typing import List

import torch
from transformers import pipeline
from categories import CATEGORIES

# ------------------- CONFIG ---------------------------------------
PARA_MODEL = "eugenesiow/bart-paraphrase"
DEVICE     = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLIMATE_WORDS = {
    # core
    "climate", "carbon", "co2", "methane", "emissions", "warming",
    "footprint", "greenhouse", "ghg", "policy", "policies",
    "education", "mitigation", "adaptation", "sustainability",
    "impact", "impacts", "temperature", "renewable", "energy",
}
STOP = {
    "the", "and", "with", "that", "have", "has", "this", "there", "are",
    "lots", "for", "to", "of", "in", "on", "a", "an", "about", "teaching",
    "kids", "kid",
}
PROFANE = {"penis", "vagina", "dick", "boob", "pussy"}
TEMPLATES = {
    "climate impacts":      "climate impacts of {phrase}",
    "greenhouse gases":     "GHG emissions from {phrase}",
    "climate mitigation":   "mitigation strategies for {phrase}",
    "climate hazards":      "hazard risks to {phrase} under climate change",
    "climate datasets":     "datasets tracking {phrase} under climate change",
    "climate models":       "model projections for {phrase}",
    "climate nature":       "ecosystem impact on {phrase}",
    "climate observations": "observed climate trends for {phrase}",
    "climate organisms":    "species response of {phrase} to climate change",
    "climate organizations":"organizations addressing {phrase}",
    "climate assets":       "financial exposure of {phrase} to climate change",
    "origins of climate problems": "drivers of emissions in {phrase}",
    "climate properties":   "thermodynamic climate properties of {phrase}",
}
# ------------------- helper pipelines ------------------------------
@lru_cache(maxsize=1)
def _paraphraser():
    return pipeline(
        "text2text-generation",
        model=PARA_MODEL,
        tokenizer=PARA_MODEL,
        device_map="auto",
        max_length=48,
    )

# ------------------- utilities ------------------------------------
def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())

def _token_overlap(a: List[str], b: List[str]) -> float:
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)

def _noun_phrase(text: str, k: int = 3) -> str:
    toks = [w for w in _tokens(text) if w not in STOP and w not in PROFANE]
    freq = collections.Counter(toks)
    return " ".join([w for w, _ in freq.most_common(k)]) or "human activities"

def _paraphrase(text: str) -> str:
    outs = _paraphraser()(
        text,
        do_sample=True,
        num_return_sequences=5,
        num_beams=5,
        temperature=0.9,
        top_p=0.9,
    )
    orig = _tokens(text)
    for o in outs:
        cand = o["generated_text"].strip().strip('"')
        if _token_overlap(orig, _tokens(cand)) < 0.6:
            return cand
    return text  # fallback

# ------------------- main API --------------------------------------
def rewrite_query(query: str, category: str) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"{category=} not recognised")

    tag   = f"<{category.replace(' ', '_')}>"
    toks  = _tokens(query)
    clim_ratio = sum(w in CLIMATE_WORDS for w in toks) / max(len(toks), 1)

    # 1. climate‑rich → paraphrase & tag
    if clim_ratio >= 0.25:
        return f"{tag} {_paraphrase(query)}"

    # 2. otherwise → template using extracted noun phrase
    phrase    = _noun_phrase(query)
    template  = TEMPLATES.get(category, "climate dimension of {phrase}")
    return f"{tag} {template.format(phrase=phrase)}"
