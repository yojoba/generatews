"""
Customizable AI news/blog post generator.

Features:
- Choose a topic to write about.
- Control how many posts to generate.
- Optionally fetch related news via GNews for grounding.
- Post results to a configurable API endpoint with its API key, or use dry-run to preview locally.
"""
import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

DEFAULT_IMAGE = "/images/blog/default.jpg"
DEFAULT_AUTHOR = "FlowBiz Team"
DEFAULT_READ_TIME = "5 min"
DEFAULT_GNEWS_MAX = 20
DEFAULT_TONE = "news"


@dataclass
class GenerationConfig:
    """Runtime options supplied by the user."""

    topic: str
    posts: int
    tone: str
    api_endpoint: Optional[str]
    api_key: Optional[str]
    gnews_key: Optional[str]
    openai_key: str
    dry_run: bool


def clean_text(text: str) -> str:
    """Remove control characters while keeping accents and spacing."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    lines = text.split("\n")
    return "\n".join(" ".join(line.split()) for line in lines).strip()


def extract_json_content(text: str) -> str:
    """Extract JSON object from a string that may contain extra text."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def parse_json_safely(json_str: str) -> Optional[Dict]:
    """Parse JSON even when wrapped with stray characters."""
    json_content = extract_json_content(json_str)
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        cleaned = clean_text(json_content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def fetch_news(query: str, max_articles: int, gnews_key: Optional[str]) -> List[Dict]:
    """Fetch related news articles for grounding if a GNews key is available."""
    if not gnews_key:
        return []

    params = {
        "q": query,
        "lang": "en",
        "country": "us",
        "max": max_articles,
        "apikey": gnews_key,
    }
    try:
        response = requests.get("https://gnews.io/api/v4/search", params=params, timeout=15)
    except requests.RequestException as exc:
        print(f"⚠️  Failed to reach GNews: {exc}")
        return []

    if response.status_code != 200:
        print(f"⚠️  GNews responded with status {response.status_code}: {response.text[:200]}")
        return []

    data = response.json()
    articles: List[Dict] = []
    for article in data.get("articles", []):
        content = BeautifulSoup(article.get("content", ""), "html.parser").get_text().strip()
        articles.append(
            {
                "title": clean_text(article.get("title", "")),
                "content": clean_text(content),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
                "image": article.get("image", DEFAULT_IMAGE),
                "source": clean_text(article.get("source", {}).get("name", "Unknown")),
            }
        )
    return articles


def build_generation_prompt(topic: str, tone: str, article: Optional[Dict]) -> str:
    """Builds a prompt using optional news grounding and requested tone."""
    if article:
        return f"""
Create a concise blog post inspired by this news article.
Topic focus: {topic}
Title: {article.get('title', '')}
Content: {article.get('content', '')}
Source URL: {article.get('url', '')}

Desired tone/style: {tone}

Return a JSON object with keys: title, excerpt, content, tags.
- excerpt: 2-3 sentence summary
- content: 3-4 paragraphs (plain text, no markdown)
- tags: 3-6 concise keywords
Respond with JSON only.
"""
    return f"""
Create a concise blog post about "{topic}" in a {tone} style.
Return a JSON object with keys: title, excerpt, content, tags.
- excerpt: 2-3 sentence summary
- content: 3-4 paragraphs (plain text, no markdown)
- tags: 3-6 concise keywords
Respond with JSON only.
"""


def generate_post(client: OpenAI, topic: str, tone: str, article: Optional[Dict]) -> Optional[Dict]:
    """Generate a single blog post, optionally grounded on an article."""
    prompt = build_generation_prompt(topic, tone, article)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional technology writer. Always respond with a single valid JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    content = parse_json_safely(response.choices[0].message.content)
    if not content:
        return None

    now = datetime.utcnow().isoformat()
    title = clean_text(content.get("title", ""))
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "post"

    return {
        "id": str(int(datetime.utcnow().timestamp() * 1000)),
        "slug": slug,
        "title": {"en": title},
        "content": {"en": clean_text(content.get("content", ""))},
        "excerpt": {"en": clean_text(content.get("excerpt", ""))},
        "image": (article or {}).get("image", DEFAULT_IMAGE),
        "date": now,
        "author": DEFAULT_AUTHOR,
        "tags": content.get("tags", []),
        "readTime": DEFAULT_READ_TIME,
        "source_url": (article or {}).get("url", ""),
        "source_name": (article or {}).get("source", ""),
    }


def post_to_api(post: Dict, endpoint: str, api_key: str) -> bool:
    """Send a generated post to the configured API endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key,
    }
    response = requests.post(endpoint, json=post, headers=headers, timeout=15)
    if response.status_code not in {200, 201}:
        print(
            f"⚠️  Failed to post '{post.get('title', {}).get('en', 'untitled')}' "
            f"(status {response.status_code}): {response.text[:200]}"
        )
        return False
    return True


def parse_args_to_config() -> GenerationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AI news/blog posts with configurable topic, count, tone, and API destination."
        ),
    )
    parser.add_argument("topic", help="Topic or query for the posts (e.g., 'AI for healthcare').")
    parser.add_argument("--posts", type=int, default=3, help="Number of posts to generate (default: 3).")
    parser.add_argument(
        "--tone",
        default=DEFAULT_TONE,
        choices=["news", "blog", "explainer", "thought-leadership"],
        help="Desired tone/style for the posts.",
    )
    parser.add_argument("--api-endpoint", dest="api_endpoint", help="Destination API endpoint URL.")
    parser.add_argument("--api-key", dest="api_key", help="API key for the destination endpoint.")
    parser.add_argument("--gnews-key", dest="gnews_key", default=os.getenv("GNEWS_API_KEY"), help="GNews API key (optional).")
    parser.add_argument(
        "--openai-key",
        dest="openai_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to OPENAI_API_KEY env variable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate posts without sending them to the API.",
    )

    args = parser.parse_args()

    if not args.openai_key:
        raise SystemExit("OPENAI_API_KEY is required (pass with --openai-key or set env variable).")

    return GenerationConfig(
        topic=args.topic,
        posts=args.posts,
        tone=args.tone,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        gnews_key=args.gnews_key,
        openai_key=args.openai_key,
        dry_run=args.dry_run,
    )


def validate_config(config: GenerationConfig) -> None:
    if config.posts < 1:
        raise SystemExit("--posts must be at least 1.")

    if bool(config.api_endpoint) ^ bool(config.api_key):
        raise SystemExit("--api-endpoint and --api-key must be provided together, or omit both for dry run.")

    if not config.gnews_key:
        print("ℹ️  No GNews API key provided; posts will be generated without live news grounding.")


def main() -> None:
    config = parse_args_to_config()
    validate_config(config)

    client = OpenAI(api_key=config.openai_key)

    destination = "dry-run (no API calls)"
    if config.api_endpoint and config.api_key and not config.dry_run:
        destination = f"POST -> {config.api_endpoint}"
    print(
        f"Generating {config.posts} {config.tone} post(s) about '{config.topic}' | "
        f"{destination}"
    )

    news_articles = fetch_news(config.topic, min(config.posts * 2, DEFAULT_GNEWS_MAX), config.gnews_key)
    if news_articles:
        print(f"Grounding with {len(news_articles)} recent article(s) from GNews.")

    generated: List[Dict] = []
    for index in range(config.posts):
        article = news_articles[index] if index < len(news_articles) else None
        post = generate_post(client, config.topic, config.tone, article)
        if not post:
            continue
        generated.append(post)

        if config.dry_run or not config.api_endpoint or not config.api_key:
            continue
        post_to_api(post, config.api_endpoint, config.api_key)

    for post in generated:
        print(json.dumps(post, indent=2, ensure_ascii=False))

    if not generated:
        raise SystemExit("No posts were generated. Check API keys and try again.")


if __name__ == "__main__":
    main()
