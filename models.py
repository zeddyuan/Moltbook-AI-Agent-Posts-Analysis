"""Data models for Moltbook API responses."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Agent:
    id: str
    name: str
    description: str = ""
    karma: int = 0
    follower_count: int = 0
    following_count: int = 0
    you_follow: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        if not data:
            return None
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            karma=data.get("karma", 0),
            follower_count=data.get("follower_count", 0),
            following_count=data.get("following_count", 0),
            you_follow=data.get("you_follow", False),
        )


@dataclass
class Submolt:
    id: str = ""
    name: str = ""
    display_name: str = ""
    description: str = ""
    subscribers: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Submolt":
        if not data:
            return None
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            subscribers=data.get("subscribers", 0),
        )


@dataclass
class Comment:
    id: str
    content: str
    parent_id: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    created_at: str = ""
    author: Optional[Agent] = None
    replies: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Comment":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            parent_id=data.get("parent_id"),
            upvotes=data.get("upvotes", 0),
            downvotes=data.get("downvotes", 0),
            created_at=data.get("created_at", ""),
            author=Agent.from_dict(data.get("author", {})),
            replies=[Comment.from_dict(r) for r in data.get("replies", [])],
        )


@dataclass
class Post:
    id: str
    title: str
    content: str = ""
    url: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    comment_count: int = 0
    created_at: str = ""
    author: Optional[Agent] = None
    submolt: Optional[Submolt] = None
    comments: list = field(default_factory=list)

    @property
    def score(self) -> int:
        return self.upvotes - self.downvotes

    @property
    def link(self) -> str:
        return f"https://www.moltbook.com/post/{self.id}"

    @classmethod
    def from_dict(cls, data: dict) -> "Post":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            url=data.get("url"),
            upvotes=data.get("upvotes", 0),
            downvotes=data.get("downvotes", 0),
            comment_count=data.get("comment_count", 0),
            created_at=data.get("created_at", ""),
            author=Agent.from_dict(data.get("author", {})),
            submolt=Submolt.from_dict(data.get("submolt", {})),
            comments=[Comment.from_dict(c) for c in data.get("comments", [])],
        )


@dataclass
class Message:
    id: str
    content: str
    sender: Optional[Agent] = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", data.get("message", "")),
            sender=Agent.from_dict(data.get("sender", {})),
            created_at=data.get("created_at", ""),
        )


@dataclass
class Conversation:
    id: str
    other_agent: Optional[Agent] = None
    last_message: Optional[Message] = None
    unread_count: int = 0
    status: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            id=data.get("id", ""),
            other_agent=Agent.from_dict(data.get("other_agent", {})),
            last_message=Message.from_dict(data.get("last_message", {})) if data.get("last_message") else None,
            unread_count=data.get("unread_count", 0),
            status=data.get("status", ""),
        )
