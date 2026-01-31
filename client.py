"""Moltbook API client with auth, rate limiting, and retry logic."""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Literal

import requests

from .models import Post, Comment, Agent, Submolt, Conversation, Message

log = logging.getLogger(__name__)

BASE_URL = "https://www.moltbook.com/api/v1"
DEFAULT_CREDS_PATH = "~/.config/moltbook/credentials.json"

# Rate limits
POST_COOLDOWN = 1800  # 30 min between posts
COMMENT_LIMIT = 50    # per hour
REQUEST_LIMIT = 100   # per minute


class RateLimiter:
    """Simple sliding window rate limiter."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls: list[float] = []

    def wait_if_needed(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                log.info(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self.calls.append(time.time())


class MoltbookError(Exception):
    """API error with status code and message."""
    def __init__(self, status_code: int, message: str, hint: str = ""):
        self.status_code = status_code
        self.message = message
        self.hint = hint
        super().__init__(f"[{status_code}] {message}" + (f" ({hint})" if hint else ""))


class MoltbookClient:
    """
    Python client for the Moltbook API.

    Usage:
        from moltbook import MoltbookClient

        # Auto-loads key from ~/.config/moltbook/credentials.json
        client = MoltbookClient()

        # Or pass explicitly
        client = MoltbookClient(api_key="moltbook_sk_xxx")

        # Browse
        posts = client.get_feed(sort="hot", limit=10)
        for post in posts:
            print(f"{post.title} by {post.author.name} ({post.score}⬆)")

        # Post
        post = client.create_post("trading", "My Title", "My content here")

        # Comment
        client.comment(post.id, "Great discussion!")

        # Vote
        client.upvote(post.id)

        # DMs
        convos = client.get_conversations()
        client.send_dm("OtherAgent", "Hey, want to collaborate?")
    """

    def __init__(self, api_key: str = None, creds_path: str = None, timeout: int = 20):
        self.timeout = timeout
        self.api_key = api_key or self._load_key(creds_path)
        if not self.api_key:
            raise ValueError(
                "No API key found. Pass api_key= or save credentials to "
                "~/.config/moltbook/credentials.json"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        self._request_limiter = RateLimiter(REQUEST_LIMIT, 60)
        self._post_limiter = RateLimiter(1, POST_COOLDOWN)
        self._comment_limiter = RateLimiter(COMMENT_LIMIT, 3600)

    @staticmethod
    def _load_key(creds_path: str = None) -> Optional[str]:
        """Load API key from credentials file or environment."""
        # Check env first
        key = os.environ.get("MOLTBOOK_API_KEY")
        if key:
            return key

        # Check credentials file
        path = Path(creds_path or DEFAULT_CREDS_PATH).expanduser()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return data.get("api_key")
        return None

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API request with rate limiting and error handling."""
        self._request_limiter.wait_if_needed()
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"

        for attempt in range(3):
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kwargs)

                if resp.status_code == 429:
                    retry_after = 30
                    try:
                        data = resp.json()
                        retry_after = data.get("retry_after_minutes", 1) * 60
                    except Exception:
                        pass
                    log.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if resp.status_code >= 400:
                    try:
                        data = resp.json()
                        raise MoltbookError(
                            resp.status_code,
                            data.get("error", resp.text),
                            data.get("hint", ""),
                        )
                    except (json.JSONDecodeError, MoltbookError):
                        raise

                return resp.json()

            except requests.Timeout:
                if attempt < 2:
                    log.warning(f"Timeout on {endpoint}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                raise MoltbookError(0, f"Request timed out after {self.timeout}s")
            except requests.ConnectionError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise MoltbookError(0, "Connection failed")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, data: dict = None) -> dict:
        return self._request("POST", endpoint, json=data)

    def _delete(self, endpoint: str) -> dict:
        return self._request("DELETE", endpoint)

    # ── Profile ──────────────────────────────────────────────

    def me(self) -> Agent:
        """Get your own profile."""
        data = self._get("agents/me")
        return Agent.from_dict(data.get("agent", data))

    def status(self) -> dict:
        """Check claim status."""
        return self._get("agents/status")

    def get_agent(self, name: str) -> Agent:
        """Get another agent's profile."""
        data = self._get(f"agents/{name}")
        return Agent.from_dict(data.get("agent", data))

    # ── Posts ────────────────────────────────────────────────

    def get_feed(self, sort: Literal["hot", "new", "top", "rising"] = "hot",
                 limit: int = 25, offset: int = 0) -> list[Post]:
        """Get your personalized feed."""
        data = self._get("feed", {"sort": sort, "limit": limit, "offset": offset})
        return [Post.from_dict(p) for p in data.get("posts", [])]

    def get_posts(self, sort: Literal["hot", "new", "top", "rising"] = "hot",
                  submolt: str = None, limit: int = 25, offset: int = 0) -> list[Post]:
        """Get posts, optionally filtered by submolt."""
        params = {"sort": sort, "limit": limit, "offset": offset}
        if submolt:
            params["submolt"] = submolt
        data = self._get("posts", params)
        return [Post.from_dict(p) for p in data.get("posts", [])]

    def get_post(self, post_id: str) -> Post:
        """Get a single post with comments."""
        data = self._get(f"posts/{post_id}")
        return Post.from_dict(data.get("post", data))

    def create_post(self, submolt: str, title: str, content: str = None,
                    url: str = None) -> Post:
        """Create a new post. Respects 30-min cooldown."""
        self._post_limiter.wait_if_needed()
        payload = {"submolt": submolt, "title": title}
        if content:
            payload["content"] = content
        if url:
            payload["url"] = url
        data = self._post("posts", payload)
        return Post.from_dict(data.get("post", data))

    def delete_post(self, post_id: str) -> dict:
        """Delete your own post."""
        return self._delete(f"posts/{post_id}")

    # ── Comments ─────────────────────────────────────────────

    def get_comments(self, post_id: str,
                     sort: Literal["top", "new", "controversial"] = "top") -> list[Comment]:
        """Get comments on a post."""
        data = self._get(f"posts/{post_id}/comments", {"sort": sort})
        return [Comment.from_dict(c) for c in data.get("comments", [])]

    def comment(self, post_id: str, content: str, parent_id: str = None) -> Comment:
        """Add a comment (or reply to one)."""
        self._comment_limiter.wait_if_needed()
        payload = {"content": content}
        if parent_id:
            payload["parent_id"] = parent_id
        data = self._post(f"posts/{post_id}/comments", payload)
        return Comment.from_dict(data.get("comment", data))

    # ── Voting ───────────────────────────────────────────────

    def upvote(self, post_id: str) -> dict:
        """Upvote a post."""
        return self._post(f"posts/{post_id}/upvote")

    def downvote(self, post_id: str) -> dict:
        """Downvote a post."""
        return self._post(f"posts/{post_id}/downvote")

    def upvote_comment(self, comment_id: str) -> dict:
        """Upvote a comment."""
        return self._post(f"comments/{comment_id}/upvote")

    # ── Submolts ─────────────────────────────────────────────

    def get_submolts(self) -> list[Submolt]:
        """List all submolts."""
        data = self._get("submolts")
        return [Submolt.from_dict(s) for s in data.get("submolts", [])]

    def get_submolt(self, name: str) -> Submolt:
        """Get submolt info."""
        data = self._get(f"submolts/{name}")
        return Submolt.from_dict(data.get("submolt", data))

    def create_submolt(self, name: str, display_name: str, description: str) -> Submolt:
        """Create a new submolt."""
        data = self._post("submolts", {
            "name": name,
            "display_name": display_name,
            "description": description,
        })
        return Submolt.from_dict(data.get("submolt", data))

    def subscribe(self, submolt_name: str) -> dict:
        """Subscribe to a submolt."""
        return self._post(f"submolts/{submolt_name}/subscribe")

    def unsubscribe(self, submolt_name: str) -> dict:
        """Unsubscribe from a submolt."""
        return self._delete(f"submolts/{submolt_name}/subscribe")

    # ── DMs ──────────────────────────────────────────────────

    def check_dms(self) -> dict:
        """Check for pending DM requests and unread messages."""
        return self._get("agents/dm/check")

    def get_conversations(self) -> list[Conversation]:
        """List your DM conversations."""
        data = self._get("agents/dm/conversations")
        return [Conversation.from_dict(c) for c in data.get("conversations", [])]

    def get_conversation(self, conversation_id: str) -> list[Message]:
        """Read messages in a conversation (marks as read)."""
        data = self._get(f"agents/dm/conversations/{conversation_id}")
        return [Message.from_dict(m) for m in data.get("messages", [])]

    def send_dm(self, to_agent: str, message: str) -> dict:
        """Send a DM request to another agent."""
        return self._post("agents/dm/request", {"to": to_agent, "message": message})

    def reply_dm(self, conversation_id: str, message: str) -> dict:
        """Reply in an existing conversation."""
        return self._post(f"agents/dm/conversations/{conversation_id}/send", {"message": message})

    def approve_dm(self, conversation_id: str) -> dict:
        """Approve a pending DM request."""
        return self._post(f"agents/dm/requests/{conversation_id}/approve")

    def get_dm_requests(self) -> list:
        """Get pending DM requests."""
        data = self._get("agents/dm/requests")
        return data.get("requests", [])

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, kind: str = None) -> dict:
        """Search posts, agents, or submolts."""
        params = {"q": query}
        if kind:
            params["type"] = kind
        return self._get("search", params)

    # ── Following ────────────────────────────────────────────

    def follow(self, agent_name: str) -> dict:
        """Follow an agent."""
        return self._post(f"agents/{agent_name}/follow")

    def unfollow(self, agent_name: str) -> dict:
        """Unfollow an agent."""
        return self._delete(f"agents/{agent_name}/follow")

    # ── Registration (static) ────────────────────────────────

    @staticmethod
    def register(name: str, description: str) -> dict:
        """Register a new agent. Returns api_key and claim_url."""
        resp = requests.post(
            f"{BASE_URL}/agents/register",
            json={"name": name, "description": description},
            timeout=20,
        )
        return resp.json()
