"""Local translation via facebook/m2m100_418M, plus a back-translation
consistency score used as a quality signal when there's no reference
translation to score against.

Model load is lazy (first call, not import time) so importing this module
in tests never pays the ~1.9GB download/load cost.
"""

from __future__ import annotations

from functools import lru_cache

from src.common import config
from src.common.similarity import text_similarity


@lru_cache(maxsize=1)
def _pipeline():
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(config.TRANSLATION_MODEL)
    model = M2M100ForConditionalGeneration.from_pretrained(config.TRANSLATION_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def _lang_code(locale: str) -> str:
    return config.LOCALE_TO_M2M100.get(locale, locale.split("-")[0].lower())


def translate(text: str, src_locale: str, tgt_lang: str = config.CANONICAL_LANG) -> str:
    """Translate `text` from src_locale's language into tgt_lang.

    num_beams=1 (greedy) rather than m2m100's beam-search default: on CPU,
    beam search multiplies decode cost by the beam count for a quality gain
    that doesn't matter much on MASSIVE's short (~7-word) utterances -- a
    live streaming agent should trade a little quality for throughput here.
    """
    import torch

    tokenizer, model, device = _pipeline()
    src_lang = _lang_code(src_locale)
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=64,
            num_beams=1,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


class TranslationResult:
    __slots__ = ("back_translation_score", "translated_text")

    def __init__(self, translated_text: str, back_translation_score: float | None) -> None:
        self.translated_text = translated_text
        self.back_translation_score = back_translation_score


def translate_with_consistency_check(text: str, src_locale: str, check_quality: bool = True) -> TranslationResult:
    """Translate to English; if the ticket wasn't already English AND
    `check_quality` is True, also translate the English result back to the
    source language and score its similarity against the original as a
    cheap quality signal (no reference translation needed -- see
    src/common/similarity.py).

    check_quality is sampled, not run for every ticket (see main.py's
    SAMPLE_BACK_TRANSLATION): the eval spec only asks for this metric "across
    a held-out sample," and it doubles translation cost (a second generate()
    call) for no benefit to tickets that were only ever going to be sampled
    out of the report anyway. Sampling it turned a ~4hr full run into a
    ~90min one with no change to what gets reported -- see LEARNING.md.
    """
    src_lang = _lang_code(src_locale)
    if src_lang == config.CANONICAL_LANG:
        return TranslationResult(translated_text=text, back_translation_score=None)

    english = translate(text, src_locale, config.CANONICAL_LANG)
    if not check_quality:
        return TranslationResult(translated_text=english, back_translation_score=None)

    roundtrip = translate(english, config.CANONICAL_LANG, src_lang)
    score = text_similarity(text, roundtrip)
    return TranslationResult(translated_text=english, back_translation_score=score)
