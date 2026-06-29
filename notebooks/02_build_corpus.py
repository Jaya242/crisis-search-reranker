"""
Day 1 Block 3: Build the 80-article corpus.

Pulls real news from RSS feeds (BBC, Guardian, NPR) and fake claims from LIAR.
Combines into data/corpus.json with full metadata.

Run with: python notebooks/02_build_corpus.py
"""

import json
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import pandas as pd

random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================

RSS_FEEDS = {
    "world": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://feeds.npr.org/1001/rss.xml",
    ],
    "tech": [
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://www.theguardian.com/uk/technology/rss",
    ],
    "business": [
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.theguardian.com/business/rss",
    ],
    "sport": [
        "http://feeds.bbci.co.uk/sport/rss.xml",
        "https://www.theguardian.com/sport/rss",
    ],
}

CRISIS_KEYWORDS = [
    "wildfire", "fire", "flood", "earthquake", "hurricane", "typhoon",
    "evacuation", "disaster", "outbreak", "epidemic", "pandemic",
    "war", "conflict", "attack", "shooting", "crisis", "emergency",
    "refugee", "casualty", "victim", "death toll", "storm", "tornado",
    "drought", "tsunami", "explosion", "crash", "killed", "injured",
    "rescue", "missing", "trapped", "collapse", "evacuated", "siege",
]

LIAR_TSV = "data/liar/train.tsv"
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context",
]

OUTPUT_PATH = "data/corpus.json"

# ============================================================================
# HELPERS
# ============================================================================

def is_crisis_text(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_source(url):
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else "unknown"


def random_recent_timestamp():
    now = datetime.now(timezone.utc)
    delta = timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
    return (now - delta).isoformat()


def random_older_timestamp():
    now = datetime.now(timezone.utc)
    delta = timedelta(days=random.randint(90, 365), hours=random.randint(0, 23))
    return (now - delta).isoformat()


# ============================================================================
# PHASE 1: Real articles from RSS
# ============================================================================

def fetch_rss_articles():
    articles = []
    for category, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            print(f"  Fetching {feed_url}...")
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = clean_text(entry.get("title", ""))
                    summary = clean_text(
                        entry.get("summary", entry.get("description", ""))
                    )
                    link = entry.get("link", "")
                    text = summary if len(summary) > 50 else f"{title}. {summary}"
                    if len(text) < 50:
                        continue
                    articles.append({
                        "title": title,
                        "text": text,
                        "source": extract_source(link),
                        "category_hint": category,
                    })
            except Exception as e:
                print(f"    SKIP (error): {e}")
    return articles


def assign_real_corpus(articles):
    crisis = [a for a in articles if is_crisis_text(a["title"] + " " + a["text"])]
    non_crisis = [
        a for a in articles if not is_crisis_text(a["title"] + " " + a["text"])
    ]
    print(f"\n  Crisis-flavored real articles: {len(crisis)}")
    print(f"  Non-crisis real articles: {len(non_crisis)}")

    random.shuffle(crisis)
    random.shuffle(non_crisis)

    crisis_picked = crisis[:20]
    non_crisis_picked = non_crisis[:20]

    if len(crisis_picked) < 20:
        deficit = 20 - len(crisis_picked)
        print(f"  WARN: padding {deficit} crisis slots from non-crisis pool")
        crisis_picked.extend(non_crisis[20:20 + deficit])
        non_crisis_picked = non_crisis[20 + deficit:40 + deficit]

    records = []
    for i, art in enumerate(crisis_picked + non_crisis_picked):
        is_crisis = i < 20
        within_half_idx = i if is_crisis else (i - 20)
        is_recent = within_half_idx < 10
        ts = random_recent_timestamp() if is_recent else random_older_timestamp()
        records.append({
            "id": f"real_{i + 1:03d}",
            "title": art["title"],
            "text": art["text"][:500],
            "source": art["source"],
            "timestamp": ts,
            "category": "crisis" if is_crisis else art["category_hint"],
            "is_crisis": is_crisis,
            "is_real": True,
        })
    return records


# ============================================================================
# PHASE 2: Fake claims from LIAR
# ============================================================================

def fetch_liar_fakes(target=40):
    df = pd.read_csv(LIAR_TSV, sep="\t", names=LIAR_COLUMNS)
    fakes = df[df["label"].isin(["pants-fire", "false"])]
    print(f"  Pants-fire/false claims available: {len(fakes)}")

    sample = fakes.sample(target, random_state=42).reset_index(drop=True)

    records = []
    for i, row in sample.iterrows():
        statement = row["statement"]
        # Force balance: first 20 = crisis, next 20 = non-crisis
        is_crisis_forced = i < 20
        within_half_idx = i if is_crisis_forced else (i - 20)
        is_recent = within_half_idx < 10
        ts = random_recent_timestamp() if is_recent else random_older_timestamp()

        records.append({
            "id": f"fake_{i + 1:03d}",
            "title": statement[:100] + ("..." if len(statement) > 100 else ""),
            "text": statement,
            "source": "unverified-news.example",
            "timestamp": ts,
            "category": "crisis" if is_crisis_forced else "political",
            "is_crisis": is_crisis_forced,
            "is_real": False,
        })
    return records


# ============================================================================
# MAIN
# ============================================================================

def main():
    Path("data").mkdir(exist_ok=True)

    print("=== Phase 1: Pull real articles from RSS feeds ===")
    rss_articles = fetch_rss_articles()
    print(f"\n  Pulled {len(rss_articles)} total articles from RSS")
    real_records = assign_real_corpus(rss_articles)

    print("\n=== Phase 2: Pull fake claims from LIAR ===")
    fake_records = fetch_liar_fakes()

    corpus = real_records + fake_records
    random.shuffle(corpus)

    print(f"\n=== Writing corpus to {OUTPUT_PATH} ===")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(corpus)} records.")

    real_count = sum(1 for r in corpus if r["is_real"])
    fake_count = len(corpus) - real_count
    crisis_count = sum(1 for r in corpus if r["is_crisis"])
    non_crisis_count = len(corpus) - crisis_count

    print("\n=== Corpus summary ===")
    print(f"  Total: {len(corpus)}")
    print(f"  Real: {real_count}, Fake: {fake_count}")
    print(f"  Crisis: {crisis_count}, Non-crisis: {non_crisis_count}")

    print("\n=== Sample records ===")
    for r in corpus[:3]:
        print(f"\n  {r['id']} (real={r['is_real']}, crisis={r['is_crisis']})")
        print(f"    Title: {r['title']}")
        print(f"    Source: {r['source']}, Date: {r['timestamp'][:10]}")
        print(f"    Text: {r['text'][:120]}...")


if __name__ == "__main__":
    main()
