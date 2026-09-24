"""Cheap text-similarity for the back-translation consistency check.

difflib's SequenceMatcher ratio (character-level longest-matching-blocks)
rather than pulling in a BLEU/chrF library: it needs no extra dependency,
requires no reference corpus, and for the short, single-sentence utterances
in this pipeline (MASSIVE intents average ~7 words) it tracks word-level
BLEU/chrF closely enough to rank "good" vs "garbled" round-trips correctly,
which is all this signal is used for. See LEARNING.md for the comparison
against chrF on a sample.
"""

from __future__ import annotations

from difflib import SequenceMatcher


def text_similarity(a: str, b: str) -> float:
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if not a_norm and not b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()
