from __future__ import annotations

from typing import List
import pandas as pd

from src.config import settings

try:
    import praw
except Exception:  # pragma: no cover
    praw = None


class RedditAPIClient:
    def __init__(self) -> None:
        self.enabled = all([
            settings.reddit_client_id,
            settings.reddit_client_secret,
            settings.reddit_user_agent,
            praw is not None,
        ])
        self.reddit = None
        if self.enabled:
            self.reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )

    def search_posts(self, query: str, subreddit: str = "all", limit: int = 50) -> pd.DataFrame:
        if not self.enabled or self.reddit is None:
            return pd.DataFrame(columns=["id", "subreddit", "title", "selftext", "created_utc", "score"])

        rows: List[dict] = []
        for post in self.reddit.subreddit(subreddit).search(query, limit=limit, sort="new"):
            rows.append({
                "id": post.id,
                "subreddit": str(post.subreddit),
                "title": post.title,
                "selftext": getattr(post, "selftext", ""),
                "created_utc": post.created_utc,
                "score": post.score,
            })
        return pd.DataFrame(rows)
