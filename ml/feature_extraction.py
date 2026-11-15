
"""
feature_extraction.py
----------------------
Extracts numerical features from product reviews.
"""

import re
import numpy as np

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

QUALITY_REASONS = {
    "quality_issue", "not_as_described", "defective"
}


def clean_tokens(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def sentiment_score(text):
    tokens = clean_tokens(text)

    if not tokens:
        return 0.0

    score = 0

    for tok in tokens:
        if tok in POSITIVE_WORDS:
            score += 1
        elif tok in NEGATIVE_WORDS:
            score -= 1

    norm = score / max(len(tokens), 1)
    return float(np.clip(norm * 4, -1, 1))


def extract_features(row):
    text = row["review_text"]
    rating = float(row["rating"])
    was_returned = int(row["was_returned"])

    return_reason = row.get("return_reason", "none")
    days_to_return = float(row.get("days_to_return", 0) or 0)
    review_burst = int(row.get("review_burst", 0) or 0)

    text_sentiment = sentiment_score(text)
    review_length = len(text.split())
    exclamations = text.count("!")

    # FIX: A quality return is possible only if the product was returned.
    quality_return = int(
        was_returned == 1 and return_reason in QUALITY_REASONS
    )

    rating_norm = (rating - 3) / 2.0

    sentiment_rating_mismatch = abs(
        text_sentiment - rating_norm
    )

    positive_after_quality_return = int(
        quality_return == 1 and
        (text_sentiment > 0.15 or rating >= 4)
    )

    fast_return = int(
        was_returned == 1 and 0 < days_to_return <= 3
    )

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
    "text_sentiment",
    "review_length",
    "exclamations",
    "rating",
    "rating_norm",
    "was_returned",
    "quality_return",
    "days_to_return",
    "fast_return",
    "review_burst",
    "sentiment_rating_mismatch",
    "positive_after_quality_return",
]
