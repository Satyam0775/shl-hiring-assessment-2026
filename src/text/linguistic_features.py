"""Lexicon/rule-based linguistic features (columns prefixed ``ling_``).

IMPORTANT: these are cheap *proxies* for grammatical control (function-word
ratios, clause-complexity markers, a few agreement/article rule flags). They are
not a grammar checker and do not detect most errors. They also cannot see
errors that Whisper silently "corrects" while transcribing.
"""
from __future__ import annotations

import re

import numpy as np

from src.text.text_features import split_sentences, tokenize

PRONOUNS = frozenset("i me my mine myself you your yours he him his she her hers it its we us our ours they them their theirs this that these those who whom whose".split())
AUX = frozenset("am is are was were be been being have has had do does did will would shall should can could may might must".split())
CONJ_COORD = frozenset("and but or so nor yet".split())
CONJ_SUBORD = frozenset("because although though while whereas since unless if when whenever whether after before until once as".split())
RELATIVE = frozenset("which who whom whose that where".split())
PREPOSITIONS = frozenset("in on at by for with about against between into through during before after above below to from up down of off over under".split())
DETERMINERS = frozenset("a an the this that these those my your his her its our their some any no every each".split())
CONTRACTION_RE = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)

AGREEMENT_PATTERNS = [
    re.compile(r"\b(?:he|she|it)\s+(?:are|were|have|do|don't)\b"),
    re.compile(r"\bi\s+(?:is|are|has)\b"),
    re.compile(r"\b(?:they|we|you)\s+(?:is|was|has|does|doesn't)\b"),
    re.compile(r"\bthere\s+(?:is|was)\s+\w+\s+(?:people|things|students|friends)\b"),
]
A_BEFORE_VOWEL = re.compile(r"\ba\s+(?!(?:uni|use|usu|uti|eu|one|once)\w*)([aeiou]\w*)\b")
AN_BEFORE_CONSONANT = re.compile(r"\ban\s+(?!(?:hour|honest|honor|honour|heir|mba|fbi|sms|x-)\w*)([bcdfgjklmnpqrstvwxyz]\w*)\b")
PASSIVE_RE = re.compile(r"\b(?:is|are|was|were|been|being|be)\s+\w+(?:ed|en)\b")
PERFECT_RE = re.compile(r"\b(?:have|has|had)\s+\w+(?:ed|en)\b")
CONDITIONAL_RE = re.compile(r"\bif\b[^.?!]*\b(?:would|could|will)\b")


def _ratio(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def extract_linguistic_features(transcript: str) -> dict[str, float]:
    transcript = transcript if isinstance(transcript, str) else ""
    low = transcript.lower()
    words = tokenize(transcript)
    n = len(words)
    sents = split_sentences(transcript)
    n_s = len(sents)
    sent_lens = [len(tokenize(s)) for s in sents]
    bigrams = list(zip(words, words[1:]))

    def count(vocab):
        return sum(1 for w in words if w in vocab)

    agreement = sum(len(p.findall(low)) for p in AGREEMENT_PATTERNS)
    articles = len(A_BEFORE_VOWEL.findall(low)) + len(AN_BEFORE_CONSONANT.findall(low))
    ending_ok = sum(1 for s in sents if s[-1] in ".!?")
    subord = count(CONJ_SUBORD)

    return {
        "ling_pronoun_ratio": _ratio(count(PRONOUNS), n),
        "ling_aux_modal_ratio": _ratio(count(AUX), n),
        "ling_coord_conj_ratio": _ratio(count(CONJ_COORD), n),
        "ling_subord_conj_ratio": _ratio(subord, n),
        "ling_relative_ratio": _ratio(count(RELATIVE), n),
        "ling_preposition_ratio": _ratio(count(PREPOSITIONS), n),
        "ling_determiner_ratio": _ratio(count(DETERMINERS), n),
        "ling_adverb_ly_ratio": _ratio(sum(1 for w in words if w.endswith("ly") and len(w) > 4), n),
        "ling_ing_ratio": _ratio(sum(1 for w in words if w.endswith("ing") and len(w) > 4), n),
        "ling_ed_ratio": _ratio(sum(1 for w in words if w.endswith("ed") and len(w) > 4), n),
        "ling_long_word_ratio": _ratio(sum(1 for w in words if len(w) >= 7), n),
        "ling_contraction_ratio": _ratio(len(CONTRACTION_RE.findall(transcript)), n),
        "ling_subord_per_sent": _ratio(subord, n_s),
        "ling_passive_count": float(len(PASSIVE_RE.findall(low))),
        "ling_perfect_count": float(len(PERFECT_RE.findall(low))),
        "ling_conditional_count": float(len(CONDITIONAL_RE.findall(low))),
        "ling_agreement_flags": float(agreement),
        "ling_agreement_flags_per100": 100.0 * _ratio(agreement, n),
        "ling_article_flags": float(articles),
        "ling_short_sent_ratio": _ratio(sum(1 for L in sent_lens if L < 4), n_s),
        "ling_long_sent_ratio": _ratio(sum(1 for L in sent_lens if L >= 20), n_s),
        "ling_unterminated_ratio": _ratio(n_s - ending_ok, n_s),
        "ling_repeated_bigram_ratio": _ratio(len(bigrams) - len(set(bigrams)), len(bigrams)),
    }
