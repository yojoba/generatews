# FlowBiz SME Blog Generator

Automation scripts that turn recent AI/automation news into bilingual (English and French) blog posts for FlowBiz. The tools pull headlines from GNews, summarize them with OpenAI, de-duplicate against the local store, and can push results to the FlowBiz API.

## Quick start
1. Install Python 3.11+ and dependencies: `pip install -r requirements.txt`.
2. Create a `.env` file alongside the scripts with `OPENAI_API_KEY=<your OpenAI API key>`.
3. Run either generator (examples below). Dry-run modes skip file writes and API calls so you can validate output first.

## Configuration

| Setting | Where it comes from | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | `.env` | Required for both scripts. |
| `GNEWS_API_KEY` | Constant inside each script | Update if your GNews key differs. |
| `API_ENDPOINT` | Constant inside each script | Defaults to `http://localhost:3000/api/blog`. |
| `API_KEY` | Constant inside each script | Used to authenticate when posting to the FlowBiz API. |
| `BLOG_POSTS_FILE` | Constant inside each script | Default: `app/src/data/blog-posts.json`. |
| `MAX_BLOG_POSTS` | Constant in the SME script | SME generator trims old posts when the cap is reached (default 15). |

> Tip: If you change any constant, modify it in the corresponding script before running.

## Scripts

### `generate_daily_sme_posts.py`
Generates up to three SME-focused posts per run, then trims the on-disk store so only the most recent posts remain.

**Workflow**
1. Loads configuration (environment variables and constants) and initializes logging to `/tmp/sme_blog_generator.log` + STDOUT.
2. Iterates through three preset SME themes, fetching fresh articles from GNews.
3. Cleans article content, removes control characters, and skips items that match existing posts by URL, title, content, or source similarity.
4. Uses OpenAI Chat Completions to produce structured HTML content (English) and translations (French).
5. Saves the post to `BLOG_POSTS_FILE`, enforces the `MAX_BLOG_POSTS` cap, and posts the payload to `API_ENDPOINT` with `API_KEY`.
6. Optional scheduler runs once on start, then every day at noon.

**Usage examples**
```bash
# Generate posts immediately (default behavior)
python3.11 generate_daily_sme_posts.py

# Run in dry-run mode (no file writes or API calls)
python3.11 generate_daily_sme_posts.py --dry-run

# Start the daily scheduler (runs once immediately, then every day at 12:00)
python3.11 generate_daily_sme_posts.py --schedule

# Remove older posts so the JSON store stays within the 15-post limit
python3.11 generate_daily_sme_posts.py --cleanup
```

### `generate_blog_posts.py`
Generates general AI news posts for any query, up to the requested number of articles.

**Workflow**
1. Reads environment variables/constants, defaulting to query `"artificial intelligence"` and a max of 15 articles.
2. Fetches articles from GNews, cleans control characters, and filters out items similar to existing posts by URL/title/content.
3. Uses OpenAI Chat Completions to draft bilingual content, excerpts, tags, and metadata.
4. Writes new posts to `BLOG_POSTS_FILE` (newest first) and optionally posts them to `API_ENDPOINT` with `API_KEY`.

**Usage examples**
```bash
# Generate posts with the default query
python3 generate_blog_posts.py

# Use a custom query and limit the number of fetched articles
python3 generate_blog_posts.py --query "small business AI" --max 10

# Preview which articles would be processed without generating posts
python3 generate_blog_posts.py --dry-run
```

## Outputs and logging
- Posts are stored in `BLOG_POSTS_FILE` (JSON with newest entries first). Each post includes English/French content, excerpts, metadata, and tags.
- `generate_daily_sme_posts.py` logs detailed progress to `/tmp/sme_blog_generator.log` and mirrors messages to STDOUT.

## Safety and data hygiene
- Control characters are stripped from article inputs and generated content before saving or posting.
- Duplicate detection prevents reprocessing the same article across both scripts.
- The SME generator enforces a `MAX_BLOG_POSTS` cap to keep the dataset bounded, cleaning up older posts automatically.
