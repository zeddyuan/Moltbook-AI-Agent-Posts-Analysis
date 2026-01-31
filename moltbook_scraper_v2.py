"""
Moltbook Data Scraper v2 - Optimized with concurrent requests.
Collects posts and comments from Moltbook forum.
"""

import requests
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE_URL = "https://www.moltbook.com/api/v1"

class MoltbookScraper:
    """Scraper for Moltbook public API with concurrent requests."""
    
    def __init__(self, timeout: int = 30, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MoltbookScraper/2.0"
        })
        self.lock = threading.Lock()
        
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the API."""
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            return {}
    
    def get_posts(self, sort: str = "new", limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get posts from the API."""
        params = {"sort": sort, "limit": limit, "offset": offset}
        data = self._get("posts", params)
        return data.get("posts", [])
    
    def get_post_with_comments(self, post_id: str) -> Dict:
        """Get a single post with comments."""
        data = self._get(f"posts/{post_id}")
        return data.get("post", data) if data else {}
    
    def get_submolts(self) -> List[Dict]:
        """Get all submolts."""
        data = self._get("submolts")
        return data.get("submolts", [])


def scrape_all_data():
    """Main function to scrape all Moltbook data."""
    scraper = MoltbookScraper(max_workers=20)
    
    all_posts = []
    all_comments = []
    all_submolts = []
    
    print("=" * 60)
    print("Starting Moltbook Data Scrape v2 (Optimized)")
    print("=" * 60)
    
    # Get submolts first
    print("\n[1] Fetching submolts...")
    submolts = scraper.get_submolts()
    all_submolts = submolts
    print(f"    Found {len(submolts)} submolts")
    
    # Get all posts with pagination
    print("\n[2] Fetching posts...")
    offset = 0
    limit = 100
    
    while True:
        posts = scraper.get_posts(sort="new", limit=limit, offset=offset)
        
        if not posts:
            break
            
        all_posts.extend(posts)
        print(f"    Got {len(all_posts)} posts...")
        
        if len(posts) < limit:
            break
            
        offset += limit
        time.sleep(0.1)
    
    print(f"\n    Total posts collected: {len(all_posts)}")
    
    # Save posts immediately
    print("\n[3] Saving posts data...")
    posts_data = {
        "scraped_at": datetime.now().isoformat(),
        "total_posts": len(all_posts),
        "posts": all_posts,
        "submolts": all_submolts
    }
    with open("/home/ubuntu/moltbook_posts.json", "w") as f:
        json.dump(posts_data, f, indent=2)
    print(f"    Posts saved to /home/ubuntu/moltbook_posts.json")
    
    # Get comments concurrently for posts that have comments
    print("\n[4] Fetching comments (concurrent)...")
    posts_with_comments = [p for p in all_posts if p.get("comment_count", 0) > 0]
    print(f"    {len(posts_with_comments)} posts have comments")
    
    def fetch_comments(post):
        post_id = post.get("id")
        if not post_id:
            return []
        
        post_details = scraper.get_post_with_comments(post_id)
        if not post_details:
            return []
        
        comments = post_details.get("comments", [])
        result = []
        
        def extract_all_comments(comment_list, post_id, post_title):
            extracted = []
            for c in comment_list:
                c["post_id"] = post_id
                c["post_title"] = post_title
                extracted.append(c)
                # Get nested replies
                replies = c.get("replies", [])
                extracted.extend(extract_all_comments(replies, post_id, post_title))
            return extracted
        
        return extract_all_comments(comments, post_id, post.get("title", ""))
    
    processed = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_comments, post): post for post in posts_with_comments}
        
        for future in as_completed(futures):
            comments = future.result()
            all_comments.extend(comments)
            processed += 1
            
            if processed % 100 == 0:
                print(f"    Processed {processed}/{len(posts_with_comments)} posts with comments...")
    
    print(f"\n    Total comments collected: {len(all_comments)}")
    
    # Save complete data
    print("\n[5] Saving complete data...")
    
    complete_data = {
        "scraped_at": datetime.now().isoformat(),
        "stats": {
            "total_posts": len(all_posts),
            "total_comments": len(all_comments),
            "total_submolts": len(all_submolts)
        },
        "submolts": all_submolts,
        "posts": all_posts,
        "comments": all_comments
    }
    
    with open("/home/ubuntu/moltbook_data.json", "w") as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"    Data saved to /home/ubuntu/moltbook_data.json")
    print("\n" + "=" * 60)
    print("Scrape Complete!")
    print(f"Posts: {len(all_posts)}")
    print(f"Comments: {len(all_comments)}")
    print(f"Submolts: {len(all_submolts)}")
    print("=" * 60)
    
    return complete_data


if __name__ == "__main__":
    scrape_all_data()
