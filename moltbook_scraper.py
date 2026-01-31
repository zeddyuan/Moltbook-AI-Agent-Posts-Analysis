"""
Moltbook Data Scraper - Collects posts and comments from Moltbook forum.
Uses direct API calls to the public endpoints.
"""

import requests
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

BASE_URL = "https://www.moltbook.com/api/v1"

class MoltbookScraper:
    """Scraper for Moltbook public API."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MoltbookScraper/1.0"
        })
        
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the API."""
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"Error {resp.status_code} for {endpoint}: {resp.text[:200]}")
                return {}
        except Exception as e:
            print(f"Exception for {endpoint}: {e}")
            return {}
    
    def get_posts(self, sort: str = "new", limit: int = 100, offset: int = 0, submolt: str = None) -> List[Dict]:
        """Get posts from the API."""
        params = {"sort": sort, "limit": limit, "offset": offset}
        if submolt:
            params["submolt"] = submolt
        data = self._get("posts", params)
        return data.get("posts", [])
    
    def get_post_details(self, post_id: str) -> Dict:
        """Get a single post with comments."""
        data = self._get(f"posts/{post_id}")
        return data.get("post", data) if data else {}
    
    def get_comments(self, post_id: str, sort: str = "top") -> List[Dict]:
        """Get comments for a post."""
        data = self._get(f"posts/{post_id}/comments", {"sort": sort})
        return data.get("comments", [])
    
    def get_submolts(self) -> List[Dict]:
        """Get all submolts."""
        data = self._get("submolts")
        return data.get("submolts", [])
    
    def get_agents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get list of agents."""
        data = self._get("agents", {"limit": limit, "offset": offset})
        return data.get("agents", [])


def scrape_all_data():
    """Main function to scrape all Moltbook data."""
    scraper = MoltbookScraper()
    
    all_posts = []
    all_comments = []
    all_submolts = []
    
    print("=" * 60)
    print("Starting Moltbook Data Scrape")
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
    total_posts = 0
    
    while True:
        print(f"    Fetching posts offset={offset}...")
        posts = scraper.get_posts(sort="new", limit=limit, offset=offset)
        
        if not posts:
            print(f"    No more posts at offset {offset}")
            break
            
        all_posts.extend(posts)
        total_posts += len(posts)
        print(f"    Got {len(posts)} posts (total: {total_posts})")
        
        if len(posts) < limit:
            break
            
        offset += limit
        time.sleep(0.5)  # Be nice to the server
    
    print(f"\n    Total posts collected: {len(all_posts)}")
    
    # Get comments for each post
    print("\n[3] Fetching comments for each post...")
    for i, post in enumerate(all_posts):
        post_id = post.get("id")
        if not post_id:
            continue
            
        if (i + 1) % 50 == 0:
            print(f"    Processing post {i+1}/{len(all_posts)}...")
        
        # Get detailed post with comments
        post_details = scraper.get_post_details(post_id)
        if post_details:
            comments = post_details.get("comments", [])
            for comment in comments:
                comment["post_id"] = post_id
                comment["post_title"] = post.get("title", "")
                all_comments.append(comment)
                
                # Also get nested replies
                def extract_replies(comment_data, post_id, post_title):
                    replies = comment_data.get("replies", [])
                    extracted = []
                    for reply in replies:
                        reply["post_id"] = post_id
                        reply["post_title"] = post_title
                        extracted.append(reply)
                        extracted.extend(extract_replies(reply, post_id, post_title))
                    return extracted
                
                all_comments.extend(extract_replies(comment, post_id, post.get("title", "")))
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"\n    Total comments collected: {len(all_comments)}")
    
    # Save data
    print("\n[4] Saving data...")
    
    data = {
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
        json.dump(data, f, indent=2)
    
    print(f"    Data saved to /home/ubuntu/moltbook_data.json")
    print("\n" + "=" * 60)
    print("Scrape Complete!")
    print(f"Posts: {len(all_posts)}")
    print(f"Comments: {len(all_comments)}")
    print(f"Submolts: {len(all_submolts)}")
    print("=" * 60)
    
    return data


if __name__ == "__main__":
    scrape_all_data()
