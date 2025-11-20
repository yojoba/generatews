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

# Load environment variables

load_dotenv(dotenv_path=".env")  # Explicit load

print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

# Configuration
DEFAULT_IMAGE = "/images/blog/default.jpg"
DEFAULT_AUTHOR = "FlowBiz Team"
DEFAULT_READ_TIME = "5 min read"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GNEWS_API_KEY = "1cceb630dfef2ea45ebc2301581cd6d2"
API_ENDPOINT = "http://localhost:3000/api/blog"
API_KEY = "b4f17b94d894a1aee38f431d7c57b3524cc827f89f4a73a56a4fbe4d95ebf97d"
BLOG_POSTS_FILE = "app/src/data/blog-posts.json"

def clean_text(text: str) -> str:
    """Clean text by removing invalid control characters while preserving accented characters."""
    if not isinstance(text, str):
        return ""
    
    try:
        # Remove control characters except newlines and tabs
        # This preserves all Unicode characters including accented letters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove any remaining invalid characters but preserve ALL Unicode characters
        # This includes accented letters (é, è, ê, à, etc.) and other special characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize whitespace but preserve newlines
        lines = text.split('\n')
        text = '\n'.join(' '.join(line.split()) for line in lines)
        
        return text.strip()
    except Exception as e:
        print(f"Warning: Error cleaning text: {e}")
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
        print(f"Warning: Error extracting JSON: {e}")
        return text

def parse_json_safely(json_str: str) -> Dict:
    """Parse JSON string safely by extracting and cleaning it first."""
    try:
        # Extract JSON content
        json_content = extract_json_content(json_str)
        
        # Try to parse the extracted content
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            # If that fails, clean and try again
            cleaned_str = clean_text(json_content)
            return json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Problematic JSON string: {json_content[:200]}...")  # Print first 200 chars for debugging
        return None
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
        return None

def save_to_json(blog_post: Dict) -> bool:
    """Save blog post to JSON file."""
    try:
        # Read existing posts
        if os.path.exists(BLOG_POSTS_FILE):
            with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"posts": []}
        
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
        
        print(f"✅ Successfully saved: {blog_post['title']['en']}")
        return True
            
    except Exception as e:
        print(f"❌ Error saving to JSON: {str(e)}")
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
            print(f"✅ Successfully posted: {blog_post['title']['en']}")
            return True
        else:
            print(f"❌ Failed to post: {blog_post['title']['en']}")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error posting to API: {str(e)}")
        return False

def get_recent_ai_news(search_query: str = "artificial intelligence", max_articles: int = 15) -> List[Dict]:
    """Fetch recent news using GNews API with custom search query."""
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": search_query,
            "lang": "en",
            "country": "us",
            "max": max_articles,
            "apikey": GNEWS_API_KEY
        }
        
        print(f"🔍 Searching for: '{search_query}'")
        print(f"Making request to GNews API with params: {params}")
        response = requests.get(url, params=params)
        
        if response.status_code == 401:
            print("Error: Invalid GNews API key. Please check your API key and try again.")
            return []
        elif response.status_code == 429:
            print("Error: API rate limit exceeded. Please try again later.")
            return []
        elif response.status_code != 200:
            print(f"Error: GNews API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
        response.raise_for_status()
        data = response.json()
        
        if not data.get("articles"):
            print("No articles found in the response.")
            return []
        
        print(f"Found {len(data.get('articles', []))} articles")
        articles = []
        for article in data.get("articles", []):
            content = BeautifulSoup(article.get("content", ""), "html.parser").get_text().strip()
            content = clean_text(content)  # Clean the content
            
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
        print(f"Error fetching news from GNews API: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def generate_blog_post_from_news(article: Dict) -> Dict:
    """Generate a blog post from a news article."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Clean the article content before using it
        clean_article = {
            'title': clean_text(article.get('title', '')),
            'content': clean_text(article.get('content', '')),
            'description': clean_text(article.get('description', '')),
            'url': article.get('url', ''),
            'source': clean_text(article.get('source', {}).get('name', 'Unknown') if isinstance(article.get('source'), dict) else str(article.get('source', 'Unknown')))
        }
        
        # Generate English content
        en_prompt = f"""Create a detailed blog post based on this news article:
        Title: {clean_article['title']}
        Description: {clean_article['description']}
        Content: {clean_article['content']}
        
        Please provide:
        1. A compelling title
        2. A 2-3 sentence excerpt
        3. A detailed article (3-4 paragraphs)
        4. 3-5 relevant tags
        
        Format the response as a valid JSON object with these keys:
        title, excerpt, content, tags
        
        Important: 
        - Do not include any control characters
        - Ensure the JSON is properly formatted
        - Do not add any text before or after the JSON object"""
        
        en_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional tech journalist. Always respond with a single, valid JSON object without any control characters or additional text."},
                {"role": "user", "content": en_prompt}
            ],
            temperature=0.7
        )
        
        en_content = parse_json_safely(en_response.choices[0].message.content)
        if not en_content:
            return None
        
        # Generate French content
        fr_prompt = f"""Translate this blog post to French:
        {json.dumps(en_content, ensure_ascii=False)}
        
        Keep the same structure but translate all text to French.
        Important: 
        - Respond with a single, valid JSON object
        - Do not include any control characters
        - Do not add any text before or after the JSON object"""
        
        fr_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Always respond with a single, valid JSON object without any control characters or additional text."},
                {"role": "user", "content": fr_prompt}
            ],
            temperature=0.7
        )
        
        fr_content = parse_json_safely(fr_response.choices[0].message.content)
        if not fr_content:
            return None
        
        # Create blog post
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
            "source_name": clean_article['source']
        }
        
        return blog_post
        
    except Exception as e:
        print(f"Error generating blog post: {str(e)}")
        return None

def similar(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_article_exists(article: Dict) -> bool:
    """Check if an article already exists in our blog posts using multiple criteria."""
    try:
        if not os.path.exists(BLOG_POSTS_FILE):
            return False
            
        with open(BLOG_POSTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check if the article URL exists in any post
        for post in data.get('posts', []):
            # Check exact URL match
            if post.get('source_url') == article['url']:
                print(f"⚠️  Duplicate detected: Same URL found in existing post '{post['title']['en']}'")
                return True
                
            # Check similar title (80% similarity threshold)
            if similar(post['title']['en'], article['title']) > 0.8:
                print(f"⚠️  Duplicate detected: Similar title found in existing post '{post['title']['en']}'")
                return True
                
            # Check if the article content is very similar (90% similarity threshold)
            if similar(post['content']['en'], article['content']) > 0.9:
                print(f"⚠️  Duplicate detected: Very similar content found in existing post '{post['title']['en']}'")
                return True
                
        return False
    except Exception as e:
        print(f"Error checking for existing article: {e}")
        return False

def main():
    """Main function to generate and save blog posts."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate blog posts from news articles with custom search queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_blog_posts.py
  python generate_blog_posts.py --query "small business AI"
  python generate_blog_posts.py --query "automation workflow" --max 10
  python generate_blog_posts.py --query "machine learning startups" --max 5
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="artificial intelligence",
        help="Search query for news articles (default: 'artificial intelligence')"
    )
    
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=15,
        help="Maximum number of articles to fetch (default: 15)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Fetch articles but don't generate blog posts (for testing)"
    )
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    print(f"🚀 Starting blog post generation...")
    print(f"📝 Search query: '{args.query}'")
    print(f"📊 Max articles: {args.max}")
    if args.dry_run:
        print("🧪 DRY RUN MODE: Will fetch articles but not generate posts")
    print()
    
    articles = get_recent_ai_news(args.query, args.max)
    
    if not articles:
        print("❌ No articles found. Exiting.")
        return
        
    print(f"\n✅ Found {len(articles)} articles for query: '{args.query}'\n")
    
    if args.dry_run:
        print("🧪 DRY RUN: Showing articles that would be processed:")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']} (Source: {article['source']})")
        print(f"\n🧪 DRY RUN: Would process {len(articles)} articles")
        return
    
    new_articles = 0
    for i, article in enumerate(articles, 1):
        print(f"📰 Processing article {i}/{len(articles)}...")
        print(f"   Title: {article['title']}")
        print(f"   Source: {article['source']}")
        
        # Check if article already exists
        if is_article_exists(article):
            print("   ⏭️  Article already exists, skipping...")
            continue
            
        print("   🆕 New article found, generating blog post...")
        blog_post = generate_blog_post_from_news(article)
        if blog_post:
            print("   ✅ Blog post generated successfully!")
            print(f"   📝 Content preview: {blog_post['content']['en'][:100]}...")
            
            if post_to_api(blog_post):
                print("   ✅ Successfully posted to API")
                new_articles += 1
            else:
                print("   ❌ Failed to post to API")
        else:
            print("   ❌ Failed to generate blog post")
            
        print()  # Add a blank line between posts
    
    print(f"\n📊 Summary:")
    print(f"   📰 Total articles processed: {len(articles)}")
    print(f"   ✅ New posts added: {new_articles}")
    print(f"   🔍 Search query used: '{args.query}'")

if __name__ == "__main__":
    main() 