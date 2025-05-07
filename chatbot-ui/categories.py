# categories.py
"""Shared category list & helper tokens for the pipeline."""

CATEGORIES = [
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

SPECIAL_TOKENS = [f"<{c.replace(' ', '_')}>" for c in CATEGORIES]
