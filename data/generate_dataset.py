"""
generate_dataset.py
--------------------
Simulates the real-world data an e-commerce platform would have:
    - a purchase
    - (optionally) a return, with a reason
    - a review left by the customer (text + star rating)

The dataset intentionally injects a *label* (genuine / suspicious) using a
rule + noise process that mimics how fake/inconsistent reviews actually show
up in the wild, e.g.:
    - 5-star glowing review text, but the product was returned for
      "quality issue" / "not as described" / "defective"  -> suspicious
    - review text sentiment strongly disagrees with the star rating -> suspicious
    - a burst of reviews from the same reviewer in a short time -> suspicious
    - normal, consistent behaviour -> genuine

This is used purely to demonstrate the pipeline end-to-end. In a real
deployment this dataset would be replaced by the platform's actual
purchase / return / review tables.
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N = 4000

PRODUCTS = [
    "Wireless Earbuds", "Yoga Mat", "Bluetooth Speaker", "Office Chair",
    "Non-stick Pan Set", "Running Shoes", "LED Desk Lamp", "Backpack",
    "Smart Watch", "Electric Kettle",
]

RETURN_REASONS = [
    "none",                # not returned
    "quality_issue",
    "not_as_described",
    "defective",
    "changed_mind",
    "wrong_size",
    "arrived_late",
]

POSITIVE_TEMPLATES = [
    "Amazing {p}! Exceeded my expectations, works perfectly.",
    "Absolutely love this {p}, best purchase this year.",
    "Excellent quality {p}, highly recommend to everyone.",
    "Five stars! The {p} is exactly as advertised and durable.",
    "Great {p}, fast delivery and superb build quality.",
    "Perfect {p}, I use it every day without any issues.",
]

NEGATIVE_TEMPLATES = [
    "Terrible {p}, stopped working within a week.",
    "Very disappointed with this {p}, quality is poor.",
    "The {p} broke after two uses, would not recommend.",
    "Not as described, the {p} feels cheap and flimsy.",
    "Waste of money, the {p} arrived damaged.",
    "Poor build quality on this {p}, returning it.",
]

NEUTRAL_TEMPLATES = [
    "The {p} is okay, does the job but nothing special.",
    "Decent {p} for the price, a few minor issues.",
    "Average {p}, met basic expectations.",
    "The {p} works fine, packaging could be better.",
]


def make_review_text(sentiment_bucket, product):
    if sentiment_bucket == "positive":
        t = random.choice(POSITIVE_TEMPLATES)
    elif sentiment_bucket == "negative":
        t = random.choice(NEGATIVE_TEMPLATES)
    else:
        t = random.choice(NEUTRAL_TEMPLATES)
    return t.format(p=product)


rows = []
for i in range(N):
    product = random.choice(PRODUCTS)
    reviewer_id = f"U{random.randint(1, 1600):05d}"

    was_returned = random.random() < 0.28  # ~28% of purchases get returned
    if was_returned:
        return_reason = random.choice(RETURN_REASONS[1:])
        days_to_return = random.randint(1, 30)
    else:
        return_reason = "none"
        days_to_return = 0

    # ---- decide the TRUE underlying behaviour pattern ----
    # quality-related return + glowing review text/rating = inconsistency signal
    quality_return = return_reason in ("quality_issue", "not_as_described", "defective")

    suspicious = False
    if quality_return and random.random() < 0.75:
        # inconsistent: product had a quality-related return but review is glowing
        sentiment_bucket = "positive"
        rating = random.choice([4, 5])
        suspicious = True
    elif quality_return:
        # honest customer: consistent negative review after a quality return
        sentiment_bucket = "negative"
        rating = random.choice([1, 2])
    elif was_returned:
        # returned for non-quality reason (size/changed mind) -> usually neutral/positive, consistent
        sentiment_bucket = random.choice(["neutral", "positive"])
        rating = {"neutral": random.choice([3, 4]), "positive": 5}[sentiment_bucket]
    else:
        # not returned -> mostly genuine, natural spread
        sentiment_bucket = random.choices(
            ["positive", "neutral", "negative"], weights=[0.55, 0.30, 0.15]
        )[0]
        rating = {"positive": random.choice([4, 5]),
                   "neutral": 3,
                   "negative": random.choice([1, 2])}[sentiment_bucket]

    # extra noise sources of suspicious behaviour, independent of returns:
    review_burst = random.random() < 0.06  # reviewer posted many reviews same day
    if review_burst:
        suspicious = suspicious or (random.random() < 0.6)

    # small amount of random label noise to keep the problem realistic (not 100% separable)
    if random.random() < 0.04:
        suspicious = not suspicious

    text = make_review_text(sentiment_bucket, product)

    rows.append({
        "review_id": f"R{i:06d}",
        "reviewer_id": reviewer_id,
        "product": product,
        "rating": rating,
        "review_text": text,
        "was_returned": int(was_returned),
        "return_reason": return_reason,
        "days_to_return": days_to_return,
        "review_burst": int(review_burst),
        "label": "suspicious" if suspicious else "genuine",
    })


from pathlib import Path

# Convert generated rows into a DataFrame
df = pd.DataFrame(rows)

# Save dataset in the current data folder
output_path = Path(__file__).resolve().parent / "reviews_dataset.csv"

df.to_csv(output_path, index=False)

print(f"Dataset saved successfully at: {output_path}")
