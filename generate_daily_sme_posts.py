#!/usr/bin/env python3
"""
🤖 FlowBiz - Générateur Automatique de Posts Quotidiens PME
Génère 3 posts quotidiens sur l'IA et l'automatisation pour les PME
Version Docker avec limitation à 15 posts maximum
"""

import json
import os
import sys
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from difflib import SequenceMatcher
import schedule
import time
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/sme_blog_generator.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configuration
DEFAULT_IMAGE = "/images/blog/default.jpg"
DEFAULT_AUTHOR = "FlowBiz Team"
DEFAULT_READ_TIME = "5 min"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GNEWS_API_KEY = "1cceb630dfef2ea45ebc2301581cd6d2"
# Post to API inside container; in production we exec inside the app container so localhost is fine
API_ENDPOINT = "http://localhost:3000/api/blog"
API_KEY = "b4f17b94d894a1aee38f431d7c57b3524cc827f89f4a73a56a4fbe4d95ebf97d"
BLOG_POSTS_FILE = "app/src/data/blog-posts.json"
MAX_BLOG_POSTS = 15  # Limitation à 15 posts maximum

# Thèmes quotidiens pour les PME
SME_DAILY_THEMES = [
    {
        "query": "SME automation benefits",
        "focus": "Avantages de l'automatisation pour les PME",
        "keywords": ["PME", "automatisation", "productivité", "efficacité"]
    },
    {
        "query": "AI tools small business",
        "focus": "Outils IA pour petites entreprises",
        "keywords": ["IA", "outils", "PME", "innovation"]
    },
    {
        "query": "digital transformation SME",
        "focus": "Transformation digitale des PME",
        "keywords": ["transformation digitale", "PME", "modernisation", "compétitivité"]
    }
]

ALLOWED_CATEGORIES = {"ai-tools", "artificial-intelligence", "digital-transformation"}

def map_category_from_theme(theme_focus: str, query: str) -> str:
    focus = (theme_focus or "").lower()
    q = (query or "").lower()
    if "tools" in q or "tools" in focus or "outils" in focus:
        return "ai-tools"
    if "digital transformation" in q or "transformation" in focus:
        return "digital-transformation"
    return "artificial-intelligence"

def clean_text(text: str) -> str:
    """Clean text by removing invalid control characters while preserving accented characters."""
    if not isinstance(text, str):
        return ""
    
    try:
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove any remaining invalid characters but preserve ALL Unicode characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace but preserve newlines
        lines = text.split('\n')
        text = '\n'.join(' '.join(line.split()) for line in lines)
        
        return text.strip()
    except Exception as e:
        logging.warning(f"Error cleaning text: {e}")
        return ""

def extract_json_content(text: str) -> str:
    """Extract JSON content from text by finding the first { and last }."""
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        return text
    except Exception as e:
        logging.warning(f"Error extracting JSON: {e}")
        return text

def parse_json_safely(json_str: str) -> Dict:
    """Parse JSON string safely by extracting and cleaning it first."""
    try:
        json_content = extract_json_content(json_str)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            cleaned_str = clean_text(json_content)
            return json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}")
        return None

def get_current_blog_posts_count() -> int:
    """Get the current number of blog posts."""
    try:
        if os.path.exists(BLOG_POSTS_FILE):
            with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data.get('posts', []))
        return 0
    except Exception as e:
        logging.error(f"Error getting blog posts count: {e}")
        return 0

def cleanup_old_posts():
    """Remove old posts to maintain MAX_BLOG_POSTS limit."""
    try:
        if not os.path.exists(BLOG_POSTS_FILE):
            return
            
        with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        posts = data.get('posts', [])
        current_count = len(posts)
        
        if current_count > MAX_BLOG_POSTS:
            # Sort by date (newest first) and keep only the latest MAX_BLOG_POSTS
            posts.sort(key=lambda x: x.get('date', ''), reverse=True)
            posts_to_keep = posts[:MAX_BLOG_POSTS]
            
            logging.info(f"🧹 Cleaning up old posts: {current_count} -> {len(posts_to_keep)}")
            
            data['posts'] = posts_to_keep
            
            with open(BLOG_POSTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"✅ Cleaned up {current_count - len(posts_to_keep)} old posts")
            
    except Exception as e:
        logging.error(f"Error cleaning up old posts: {e}")

def save_to_json(blog_post: Dict) -> bool:
    """Save blog post to JSON file with cleanup."""
    try:
        # Read existing posts
        if os.path.exists(BLOG_POSTS_FILE):
            with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"posts": []}
        
        # Check if we're at the limit
        current_count = len(data.get('posts', []))
        if current_count >= MAX_BLOG_POSTS:
            logging.warning(f"⚠️  Blog posts limit reached ({MAX_BLOG_POSTS}). Cleaning up old posts...")
            cleanup_old_posts()
            
            # Re-read after cleanup
            with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Clean the content before saving
        blog_post['content']['en'] = clean_text(blog_post['content']['en'])
        blog_post['content']['fr'] = clean_text(blog_post['content']['fr'])
        blog_post['excerpt']['en'] = clean_text(blog_post['excerpt']['en'])
        blog_post['excerpt']['fr'] = clean_text(blog_post['excerpt']['fr'])
        
        # Add new post
        data['posts'].append(blog_post)
        
        # Sort posts by date in descending order (newest first)
        data['posts'].sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Write back to file
        with open(BLOG_POSTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Successfully saved: {blog_post['title']['en']}")
        logging.info(f"📊 Total posts: {len(data['posts'])}/{MAX_BLOG_POSTS}")
        return True
            
    except Exception as e:
        logging.error(f"❌ Error saving to JSON: {str(e)}")
        return False

def post_to_api(blog_post: Dict) -> bool:
    """Send blog post to API endpoint."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": API_KEY
        }
        
        # Clean the content before sending
        blog_post['content']['en'] = clean_text(blog_post['content']['en'])
        blog_post['content']['fr'] = clean_text(blog_post['content']['fr'])
        blog_post['excerpt']['en'] = clean_text(blog_post['excerpt']['en'])
        blog_post['excerpt']['fr'] = clean_text(blog_post['excerpt']['fr'])
        
        response = requests.post(
            API_ENDPOINT,
            json=blog_post,
            headers=headers
        )
        
        if response.status_code in [200, 201]:
            logging.info(f"✅ Successfully posted: {blog_post['title']['en']}")
            return True
        else:
            logging.error(f"❌ Failed to post: {blog_post['title']['en']}")
            logging.error(f"Status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error posting to API: {str(e)}")
        return False

def get_sme_news(search_query: str, max_articles: int = 10) -> List[Dict]:
    """Fetch recent SME-focused news using GNews API."""
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": search_query,
            "lang": "en",
            "country": "us",
            "max": max_articles,
            "apikey": GNEWS_API_KEY
        }
        
        logging.info(f"🔍 Searching for SME news: '{search_query}'")
        response = requests.get(url, params=params)
        
        if response.status_code == 401:
            logging.error("Error: Invalid GNews API key.")
            return []
        elif response.status_code == 429:
            logging.error("Error: API rate limit exceeded.")
            return []
        elif response.status_code != 200:
            logging.error(f"Error: GNews API returned status code {response.status_code}")
            return []
            
        response.raise_for_status()
        data = response.json()
        
        if not data.get("articles"):
            logging.warning("No articles found in the response.")
            return []
        
        logging.info(f"Found {len(data.get('articles', []))} articles")
        articles = []
        for article in data.get("articles", []):
            content = BeautifulSoup(article.get("content", ""), "html.parser").get_text().strip()
            content = clean_text(content)
            
            articles.append({
                'title': clean_text(article.get('title', '')),
                'content': content,
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'image': article.get('image', DEFAULT_IMAGE),
                'source': clean_text(article.get('source', {}).get('name', 'Unknown'))
            })
        
        return articles
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news from GNews API: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []

def generate_sme_blog_post(article: Dict, theme_focus: str) -> Dict:
    """Generate a SME-focused blog post from a news article."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Clean the article content
        clean_article = {
            'title': clean_text(article.get('title', '')),
            'content': clean_text(article.get('content', '')),
            'url': article.get('url', ''),
            'source': clean_text(article.get('source', {}).get('name', 'Unknown') if isinstance(article.get('source'), dict) else str(article.get('source', 'Unknown')))
        }
        
        # Generate English content with SME focus (HTML content with headings and lists for better rendering)
        en_prompt = f"""
Create a structured blog post for SMEs (Small and Medium Enterprises) based on this news article.

Theme Focus: {theme_focus}
Source Title: {clean_article['title']}
Source Content: {clean_article['content']}
Source URL: {clean_article['url']}

Requirements:
- Audience: SME owners and managers (non-technical tone, practical value)
- Output JSON with keys: title, excerpt, content, tags
- title: compelling, SME-focused
- excerpt: 2-3 sentences summary
- content: VALID HTML string only (no Markdown). Structure with multiple <h2> sections, optional <h3> subsections, <ul>/<ol> lists, <blockquote> for a key insight, and a short concluding section.
- Include sections like:
  <h2>Why it matters for SMEs</h2> (bullet points)
  <h2>Practical steps</h2> (numbered list)
  <h2>Tools and tips</h2> (bulleted list)
  <h2>Conclusion</h2>
- Include one "Source" line at the end linking to the article: <p><strong>Source:</strong> <a href=\"{clean_article['url']}\" target=\"_blank\" rel=\"noopener\">{clean_article['source']}</a></p>
- Use semantic HTML only: <h2>, <h3>, <p>, <ul>, <ol>, <li>, <blockquote>, <em>, <strong>, <a>, <code>.
- No <script> or external embeds. No images.
- Keep it concise (600-900 words), scannable, and action-oriented.

Return ONLY a single valid JSON object.
"""
        
        en_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a business consultant specializing in helping SMEs adopt technology. Always respond with a single, valid JSON object focused on practical SME benefits."},
                {"role": "user", "content": en_prompt}
            ],
            temperature=0.7
        )
        
        en_content = parse_json_safely(en_response.choices[0].message.content)
        if not en_content:
            return None
        
        # Generate French content
        fr_prompt = f"""
Translate this SME-focused blog post to French while preserving ALL HTML tags and structure.

JSON to translate:
{json.dumps(en_content, ensure_ascii=False)}

Instructions:
- Translate only text content; keep HTML tags and structure (e.g., <h2>, <h3>, <p>, <ul>, <li>, <blockquote>, <a>, <strong>, <em>, <code>) intact.
- Keep the tone professional, practical, and adapted to PME.
- Return ONLY a single, valid JSON object with the same keys (title, excerpt, content, tags).
"""
        
        fr_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator specializing in business content for French-speaking SMEs. Always respond with a single, valid JSON object."},
                {"role": "user", "content": fr_prompt}
            ],
            temperature=0.7
        )
        
        fr_content = parse_json_safely(fr_response.choices[0].message.content)
        if not fr_content:
            return None
        
        # Create blog post
        category = map_category_from_theme(theme_focus, article.get('query', ''))
        blog_post = {
            "id": str(int(datetime.now().timestamp() * 1000)),
            "slug": re.sub(r'[^a-z0-9]+', '-', clean_article['title'].lower()),
            "title": {
                "en": en_content['title'],
                "fr": fr_content['title']
            },
            "content": {
                "en": en_content['content'],
                "fr": fr_content['content']
            },
            "excerpt": {
                "en": en_content['excerpt'],
                "fr": fr_content['excerpt']
            },
            "image": article.get('image', DEFAULT_IMAGE),
            "date": datetime.now().isoformat(),
            "author": DEFAULT_AUTHOR,
            "tags": en_content['tags'],
            "readTime": DEFAULT_READ_TIME,
            "source_url": clean_article['url'],
            "source_name": clean_article['source'],
            "category": category if category in ALLOWED_CATEGORIES else "artificial-intelligence",
            "theme_focus": theme_focus
        }
        
        return blog_post
        
    except Exception as e:
        logging.error(f"Error generating SME blog post: {str(e)}")
        return None

def similar(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_article_exists(article: Dict) -> bool:
    """Check if an article already exists in our blog posts with enhanced duplicate detection."""
    try:
        if not os.path.exists(BLOG_POSTS_FILE):
            return False
            
        with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for post in data.get('posts', []):
            # Check exact URL match
            if post.get('source_url') == article['url']:
                logging.info(f"⚠️  Duplicate detected: Same URL found in existing post '{post['title']['en']}'")
                return True
                
            # Check similar title (75% similarity threshold - more strict)
            if similar(post['title']['en'], article['title']) > 0.75:
                logging.info(f"⚠️  Duplicate detected: Similar title found in existing post '{post['title']['en']}'")
                return True
                
            # Check if the article content is very similar (85% similarity threshold)
            if similar(post['content']['en'], article['content']) > 0.85:
                logging.info(f"⚠️  Duplicate detected: Very similar content found in existing post '{post['title']['en']}'")
                return True
                
            # Check source name similarity
            if similar(post.get('source_name', ''), article.get('source', '')) > 0.9:
                logging.info(f"⚠️  Duplicate detected: Same source found in existing post '{post['title']['en']}'")
                return True
                
        return False
    except Exception as e:
        logging.error(f"Error checking for existing article: {e}")
        return False

def generate_daily_sme_posts():
    """Generate 3 daily SME-focused blog posts with limit enforcement."""
    if not OPENAI_API_KEY:
        logging.error("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Check current blog posts count
    current_count = get_current_blog_posts_count()
    logging.info(f"📊 Current blog posts: {current_count}/{MAX_BLOG_POSTS}")
    
    if current_count >= MAX_BLOG_POSTS:
        logging.warning(f"⚠️  Blog posts limit reached ({MAX_BLOG_POSTS}). Cleaning up old posts...")
        cleanup_old_posts()
        current_count = get_current_blog_posts_count()
        logging.info(f"📊 After cleanup: {current_count}/{MAX_BLOG_POSTS}")
    
    logging.info("🚀 Starting daily SME blog post generation...")
    logging.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_generated = 0
    
    for i, theme in enumerate(SME_DAILY_THEMES, 1):
        # Check if we've reached the limit
        if get_current_blog_posts_count() >= MAX_BLOG_POSTS:
            logging.warning(f"⚠️  Blog posts limit reached ({MAX_BLOG_POSTS}). Stopping generation.")
            break
            
        logging.info(f"\n📝 Processing theme {i}/3: {theme['focus']}")
        logging.info(f"🔍 Search query: '{theme['query']}'")
        
        # Get articles for this theme
        articles = get_sme_news(theme['query'], max_articles=5)
        
        if not articles:
            logging.warning(f"❌ No articles found for theme: {theme['focus']}")
            continue
        
        # Try to generate a post from the first non-duplicate article
        for article in articles:
            if is_article_exists(article):
                logging.info(f"⏭️  Article already exists, trying next...")
                continue
            
            logging.info(f"🆕 New article found: {article['title']}")
            blog_post = generate_sme_blog_post(article, theme['focus'])
            
            if blog_post:
                logging.info(f"✅ Blog post generated successfully!")
                logging.info(f"📝 Title: {blog_post['title']['en']}")
                
                # Save to JSON
                if save_to_json(blog_post):
                    logging.info("✅ Successfully saved to JSON")
                    
                    # Post to API
                    if post_to_api(blog_post):
                        logging.info("✅ Successfully posted to API")
                        total_generated += 1
                        break  # Move to next theme
                    else:
                        logging.error("❌ Failed to post to API")
                else:
                    logging.error("❌ Failed to save to JSON")
            else:
                logging.error("❌ Failed to generate blog post")
        
        # Add delay between themes to avoid rate limiting
        time.sleep(2)
    
    logging.info(f"\n📊 Daily Summary:")
    logging.info(f"   📝 Themes processed: {len(SME_DAILY_THEMES)}")
    logging.info(f"   ✅ Posts generated: {total_generated}")
    logging.info(f"   📊 Total posts: {get_current_blog_posts_count()}/{MAX_BLOG_POSTS}")
    logging.info(f"   📅 Date: {datetime.now().strftime('%Y-%m-%d')}")

def run_scheduler():
    """Run the scheduler to generate posts daily at noon."""
    logging.info("⏰ Starting daily scheduler...")
    logging.info("📅 Posts will be generated daily at 12:00 (noon)")
    
    # Schedule daily posts at noon
    schedule.every().day.at("12:00").do(generate_daily_sme_posts)
    
    # Run once immediately for testing
    logging.info("🚀 Running initial generation...")
    generate_daily_sme_posts()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def main():
    """Main function with command line options."""
    parser = argparse.ArgumentParser(
        description="Generate daily SME-focused blog posts on AI & Automation (Docker-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.11 generate_daily_sme_posts.py                    # Generate posts now
  python3.11 generate_daily_sme_posts.py --schedule         # Start daily scheduler
  python3.11 generate_daily_sme_posts.py --dry-run          # Test without saving
  python3.11 generate_daily_sme_posts.py --cleanup          # Clean up old posts
        """
    )
    
    parser.add_argument(
        "--schedule", "-s",
        action="store_true",
        help="Start the daily scheduler (posts at noon)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Test the generation without saving posts"
    )
    
    parser.add_argument(
        "--cleanup", "-c",
        action="store_true",
        help="Clean up old posts to maintain limit"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        logging.info("🧹 Cleaning up old posts...")
        cleanup_old_posts()
        logging.info(f"📊 Current posts: {get_current_blog_posts_count()}/{MAX_BLOG_POSTS}")
        return
    
    if args.dry_run:
        logging.info("🧪 DRY RUN MODE: Testing generation without saving")
        # Modify functions to not save for dry run
        global save_to_json, post_to_api
        original_save = save_to_json
        original_post = post_to_api
        
        def dry_save(blog_post):
            logging.info(f"🧪 DRY RUN: Would save: {blog_post['title']['en']}")
            return True
        
        def dry_post(blog_post):
            logging.info(f"🧪 DRY RUN: Would post: {blog_post['title']['en']}")
            return True
        
        save_to_json = dry_save
        post_to_api = dry_post
    
    if args.schedule:
        run_scheduler()
    else:
        generate_daily_sme_posts()

if __name__ == "__main__":
    main() 