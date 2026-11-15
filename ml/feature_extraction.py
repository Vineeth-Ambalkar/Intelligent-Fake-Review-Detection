"""
feature_extraction.py
----------------------
Turns raw (review_text, rating, return info) into the numeric feature
vector the ML model consumes. Two feature families, mirroring the
project's Workflow diagram:

  1. TEXT / NLP FEATURES  - a lightweight lexicon-based sentiment engine
     (stands in for NLTK/spaCy/VADER in this offline demo; swapping in
     a real NLTK VADER or spaCy pipeline is a drop-in replacement -
     see the "Future Enhancements" note about BERT in the README).

  2. BEHAVIOURAL FEATURES - purchase/return signals: was it returned,
     was the return reason quality-related, how quickly, review length,
     reviewer burst activity, and critically the *mismatch* between
     text sentiment and the return/rating behaviour - this mismatch is
     the core "positive review after a quality-related return" signal
     described in the problem statement.
"""

import re
import numpy as np

# ---------------------------------------------------------------
# 1. Lightweight lexicon-based sentiment analyzer
# ---------------------------------------------------------------
POSITIVE_WORDS = {
    "amazing", "love", "excellent", "great", "perfect", "best", "superb",
    "recommend", "durable", "fast", "exceeded", "expectations", "happy",
    "good", "nice", "awesome", "fantastic", "smooth", "reliable",
}
NEGATIVE_WORDS = {
    "terrible", "disappointed", "poor", "broke", "waste", "damaged",
    "cheap", "flimsy", "bad", "worst", "defective", "stopped", "failed",
    "returning", "issue", "issues", "not", "never", "junk",
}
NEGATIONS = {"not", "no", "never", "n't"}


def clean_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def sentiment_score(text):
    """Returns a score in [-1, 1]: -1 very negative, +1 very positive."""
    tokens = clean_tokens(text)
    if not tokens:
        return 0.0
    score = 0
    for i, tok in enumerate(tokens):
        if tok in POSITIVE_WORDS:
            score += 1
        elif tok in NEGATIVE_WORDS:
            score -= 1
    norm = score / max(len(tokens), 1)
    return float(np.clip(norm * 4, -1, 1))  # scale up, then clip


QUALITY_REASONS = {"quality_issue", "not_as_described", "defective"}


def extract_features(row):
    """
    row: dict-like with keys
        review_text, rating, was_returned, return_reason,
        days_to_return, review_burst (optional, default 0)
    Returns: dict of numeric features
    """
    text = row["review_text"]
    rating = float(row["rating"])
    was_returned = int(row["was_returned"])
    return_reason = row.get("return_reason", "none")
    days_to_return = float(row.get("days_to_return", 0) or 0)
    review_burst = int(row.get("review_burst", 0) or 0)

    text_sentiment = sentiment_score(text)          # NLP feature
    review_length = len(text.split())                 # NLP feature
    exclamations = text.count("!")                     # NLP feature

    quality_return = int(return_reason in QUALITY_REASONS)

    # normalized rating in [-1, 1] to compare directly with sentiment
    rating_norm = (rating - 3) / 2.0

    # core inconsistency signal: rating/text positive AND a quality-related return
    sentiment_rating_mismatch = abs(text_sentiment - rating_norm)
    positive_after_quality_return = int(
        quality_return == 1 and (text_sentiment > 0.15 or rating >= 4)
    )

    fast_return = int(was_returned == 1 and 0 < days_to_return <= 3)

    return {
        "text_sentiment": text_sentiment,
        "review_length": review_length,
        "exclamations": exclamations,
        "rating": rating,
        "rating_norm": rating_norm,
        "was_returned": was_returned,
        "quality_return": quality_return,
        "days_to_return": days_to_return,
        "fast_return": fast_return,
        "review_burst": review_burst,
        "sentiment_rating_mismatch": sentiment_rating_mismatch,
        "positive_after_quality_return": positive_after_quality_return,
    }


FEATURE_ORDER = [
    "text_sentiment", "review_length", "exclamations", "rating",
    "rating_norm", "was_returned", "quality_return", "days_to_return",
    "fast_return", "review_burst", "sentiment_rating_mismatch",
    "positive_after_quality_return",
]
