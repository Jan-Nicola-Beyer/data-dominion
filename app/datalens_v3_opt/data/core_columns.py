"""
Core column registry for social-media datasets.

CORE_COLUMNS maps canonical field names to metadata (category, required flag,
description, dtype hint, and known column-name aliases).  CATEGORY_COLORS
assigns a display colour to each category for the UI.
"""

CORE_COLUMNS: dict = {
    # ── identity ──────────────────────────────────────────────────────────────
    "post_id": {
        "category": "identity", "required": True,
        "description": "Unique identifier of the post or content item",
        "dtype_hint": "id",
        "aliases": ["id", "post_id", "tweet_id", "status_id", "item_id", "media_id",
                    "content_id", "message_id", "video_id", "reel_id", "tiktok_id",
                    "object_id", "guid", "uid", "record_id", "external_id", "pid"],
    },
    "platform": {
        "category": "identity", "required": True,
        "description": "Social media platform such as Twitter Instagram TikTok",
        "dtype_hint": "categorical",
        "aliases": ["platform", "source", "network", "site", "channel", "medium",
                    "service", "social_network", "origin", "data_source", "network_name"],
    },
    "post_url": {
        "category": "identity", "required": False,
        "description": "URL or permalink linking to the post",
        "dtype_hint": "url",
        "aliases": ["url", "link", "permalink", "post_url", "post_link", "tweet_url",
                    "media_url", "content_url", "share_url", "source_url", "mention_url"],
    },
    # ── content ───────────────────────────────────────────────────────────────
    "content_text": {
        "category": "content", "required": True,
        "description": "Full text caption or body of the post",
        "dtype_hint": "text",
        "aliases": ["text", "full_text", "fulltext", "content", "message", "body", "post",
                    "tweet", "caption", "description", "post_text", "tweet_text",
                    "content_text", "full text", "mention_snippet", "video_description",
                    "post content", "original text", "title", "snippet"],
    },
    "language": {
        "category": "content", "required": False,
        "description": "Language code or name of the post",
        "dtype_hint": "categorical",
        "aliases": ["language", "lang", "locale", "content_language", "post_language",
                    "detected_language", "language_code", "region_language"],
    },
    "hashtags": {
        "category": "content", "required": False,
        "description": "Hashtags or topics used in the post",
        "dtype_hint": "string",
        "aliases": ["hashtags", "tags", "hashtag", "topics", "keywords", "hash_tags",
                    "hashtag_list", "post_tags", "hashtag_names", "entities_hashtags"],
    },
    "media_type": {
        "category": "content", "required": False,
        "description": "Type of media attached photo video reel story",
        "dtype_hint": "categorical",
        "aliases": ["media_type", "content_type", "post_type", "type", "format",
                    "attachment_type"],
    },
    # ── time ──────────────────────────────────────────────────────────────────
    "created_at": {
        "category": "time", "required": True,
        "description": "Timestamp when the post was published",
        "dtype_hint": "datetime",
        "aliases": ["created_at", "date", "datetime", "timestamp", "published_at",
                    "post_date", "post_time", "date_posted", "published", "created",
                    "date_created", "posted_at", "publication_date", "time_posted",
                    "date_time", "pubdate", "date published", "create_time",
                    "published date", "date time"],
    },
    "collected_at": {
        "category": "time", "required": False,
        "description": "Timestamp when data was scraped or collected",
        "dtype_hint": "datetime",
        "aliases": ["collected_at", "retrieved_at", "scraped_at", "crawled_at",
                    "collection_date", "fetch_time", "ingested_at", "added", "date_added",
                    "pull_date"],
    },
    # ── author ────────────────────────────────────────────────────────────────
    "author_id": {
        "category": "author", "required": False,
        "description": "Unique numeric or string identifier of the author account",
        "dtype_hint": "id",
        "aliases": ["author_id", "user_id", "userid", "account_id", "creator_id",
                    "profile_id", "sender_id", "owner_id", "from_id", "author id"],
    },
    "author_username": {
        "category": "author", "required": False,
        "description": "Username handle or screen name of the author",
        "dtype_hint": "string",
        "aliases": ["username", "handle", "screen_name", "user", "author", "from",
                    "account", "creator", "user_name", "author_name", "author_username",
                    "twitter_handle", "instagram_handle", "tiktok_username",
                    "owner_username", "profile_name"],
    },
    "author_display_name": {
        "category": "author", "required": False,
        "description": "Display name or full name shown on the profile",
        "dtype_hint": "string",
        "aliases": ["display_name", "full_name", "name", "real_name", "profile_name",
                    "account_name", "author_display_name", "display name"],
    },
    "author_verified": {
        "category": "author", "required": False,
        "description": "Whether the author account is verified or has blue check",
        "dtype_hint": "boolean",
        "aliases": ["verified", "is_verified", "account_verified", "blue_check",
                    "author_verified", "is_stem_verified", "verified_account"],
    },
    "author_followers_count": {
        "category": "author", "required": False,
        "description": "Number of followers the author has",
        "dtype_hint": "integer",
        "aliases": ["followers", "followers_count", "follower_count", "num_followers",
                    "audience_size", "subscribers", "fan_count", "twitter_followers",
                    "owner_followers_count"],
    },
    "author_following_count": {
        "category": "author", "required": False,
        "description": "Number of accounts the author is following",
        "dtype_hint": "integer",
        "aliases": ["following", "following_count", "friends_count", "num_following",
                    "twitter_following"],
    },
    "author_posts_count": {
        "category": "author", "required": False,
        "description": "Total number of posts published by the author",
        "dtype_hint": "integer",
        "aliases": ["posts_count", "tweet_count", "media_count", "post_count",
                    "owner_media_count", "status_count", "tweet_count_total"],
    },
    "author_location": {
        "category": "author", "required": False,
        "description": "Geographic location of the author",
        "dtype_hint": "string",
        "aliases": ["location", "author_location", "user_location", "country",
                    "region", "city", "geo", "place"],
    },
    # ── engagement ────────────────────────────────────────────────────────────
    "like_count": {
        "category": "engagement", "required": False,
        "description": "Number of likes favorites or hearts",
        "dtype_hint": "integer",
        "aliases": ["likes", "like_count", "favorites", "hearts", "favorite_count",
                    "reactions", "thumbs_up", "num_likes", "digg_count", "favorites_count",
                    "facebook_interactions"],
    },
    "comment_count": {
        "category": "engagement", "required": False,
        "description": "Number of comments replies or responses",
        "dtype_hint": "integer",
        "aliases": ["comments", "comment_count", "replies", "reply_count", "num_comments",
                    "responses", "comments_count", "story_replies"],
    },
    "share_count": {
        "category": "engagement", "required": False,
        "description": "Number of shares retweets reposts",
        "dtype_hint": "integer",
        "aliases": ["shares", "share_count", "retweets", "reposts", "repost_count",
                    "rt_count", "reshares", "num_shares", "retweet_count"],
    },
    "view_count": {
        "category": "engagement", "required": False,
        "description": "Number of views impressions or plays",
        "dtype_hint": "integer",
        "aliases": ["views", "view_count", "impressions", "impression_count",
                    "reach", "plays", "video_views", "num_views",
                    "ig_reels_video_view_total_time", "social_interactions"],
    },
    "engagement_total": {
        "category": "engagement", "required": False,
        "description": "Total engagement interactions across all types",
        "dtype_hint": "integer",
        "aliases": ["engagement", "total_engagement", "engagements", "interactions",
                    "engagement_count", "total_interactions"],
    },
    "engagement_rate": {
        "category": "engagement", "required": False,
        "description": "Engagement rate as percentage or ratio",
        "dtype_hint": "float",
        "aliases": ["engagement_rate", "eng_rate", "rate", "engagement_ratio"],
    },
    "quote_count": {
        "category": "engagement", "required": False,
        "description": "Number of quote tweets or quote posts",
        "dtype_hint": "integer",
        "aliases": ["quotes", "quote_count", "quote_tweets", "qt_count"],
    },
    # ── context ───────────────────────────────────────────────────────────────
    "in_reply_to_id": {
        "category": "context", "required": False,
        "description": "ID of the post this is a reply to",
        "dtype_hint": "id",
        "aliases": ["in_reply_to", "in_reply_to_id", "reply_to_id", "parent_id",
                    "in_reply_to_status_id", "replied_to", "thread_id"],
    },
    "is_reply": {
        "category": "context", "required": False,
        "description": "Boolean flag whether this post is a reply",
        "dtype_hint": "boolean",
        "aliases": ["is_reply", "reply", "is_comment", "is_response", "thread_entry_type"],
    },
    "is_repost": {
        "category": "context", "required": False,
        "description": "Boolean flag whether this is a retweet repost or reshare",
        "dtype_hint": "boolean",
        "aliases": ["is_repost", "is_retweet", "retweet", "retweeted", "is_rt",
                    "is_reshare", "referenced_tweets_type"],
    },
    "mentioned_handles": {
        "category": "context", "required": False,
        "description": "User handles or accounts mentioned in the post",
        "dtype_hint": "string",
        "aliases": ["mentions", "mentioned_users", "user_mentions", "mentioned_handles",
                    "tagged_users", "mentioned_accounts", "tagged"],
    },
    "sentiment": {
        "category": "context", "required": False,
        "description": "Sentiment classification positive negative neutral",
        "dtype_hint": "categorical",
        "aliases": ["sentiment", "tone", "polarity", "sentiment_label", "emotion"],
    },
}

CATEGORY_COLORS: dict = {
    "identity":   "#3b82f6",
    "content":    "#8b5cf6",
    "time":       "#14b8a6",
    "author":     "#f97316",
    "engagement": "#22c55e",
    "context":    "#ec4899",
}
