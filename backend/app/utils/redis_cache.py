"""
Redis Cache Client for NewsIntel AI
=====================================
Real Redis integration replacing the mock cache.

Architecture:
    User Query → Generate Cache Key → Redis GET → Cache Hit? → Return Cached
                                                 ↓ Cache Miss
                                                 Policy Agent → ... → Final Response → Redis SET

Cache Key Format:
    newsintel:query:<sha256_of_normalized_query>

TTL:
    Configurable via CACHE_TTL_SECONDS environment variable (default: 3600 seconds = 1 hour)

Error Handling:
    Redis failures are logged as warnings and the workflow continues normally.
    A Redis error will NEVER crash the application.

Cache Hit Rate:
    Tracked in-process via _cache_hits and _cache_misses counters.
    Rate = cache_hits / (cache_hits + cache_misses)
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB   = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))   # 1 hour default
CACHE_KEY_PREFIX  = "newsintel:query:"

# ---------------------------------------------------------------------------
# In-process hit/miss counters (reset on process restart)
# ---------------------------------------------------------------------------
_cache_hits   = 0
_cache_misses = 0


def get_cache_hit_rate() -> float:
    """
    Dynamically calculate cache hit rate from actual hit/miss events.
    Returns 0.0 if no requests have been processed yet.
    """
    total = _cache_hits + _cache_misses
    if total == 0:
        return 0.0
    return round(_cache_hits / total, 4)


def reset_cache_counters():
    """Reset hit/miss counters. Useful for testing."""
    global _cache_hits, _cache_misses
    _cache_hits   = 0
    _cache_misses = 0


# ---------------------------------------------------------------------------
# Cache Key Generation
# ---------------------------------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Normalize a query string for deterministic cache key generation.
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces into one
    - Remove punctuation that doesn't affect meaning
    """
    if not query:
        return ""
    normalized = query.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def generate_cache_key(query: str) -> str:
    """
    Generate a deterministic cache key from the normalized query.

    Format: newsintel:query:<sha256_hex_of_normalized_query>

    The same query (after normalization) always produces the same key.
    Random values are never used.
    """
    normalized = normalize_query(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Redis Client (lazy singleton)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_failed = False

def _get_redis_client():
    """
    Return a Redis client, creating it lazily on first call.
    Returns None if Redis is unavailable (connection refused, etc.).
    The application continues normally on None without repeating long timeouts.
    """
    global _redis_client, _redis_failed
    if _redis_client is not None:
        return _redis_client if _redis_client is not False else None
    if _redis_failed:
        return None

    try:
        import redis
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            decode_responses=True,
        )
        # Verify connection
        client.ping()
        _redis_client = client
        logger.info(f"[Cache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return _redis_client
    except Exception as e:
        _redis_failed = True
        _redis_client = False
        logger.warning(f"[Cache] Redis unavailable ({e}). Caching disabled — workflow continues normally.")
        return None



# ---------------------------------------------------------------------------
# Public Cache API
# ---------------------------------------------------------------------------

def cache_get(query: str):
    """
    Attempt to retrieve a cached response for the given query.

    Returns:
        str  -- cached response if a cache hit
        None -- cache miss or Redis unavailable
    """
    global _cache_hits, _cache_misses
    client = _get_redis_client()
    if client is None:
        _cache_misses += 1
        return None
    key = generate_cache_key(query)
    try:
        value = client.get(key)
        if value:
            _cache_hits += 1
            logger.info(f"[Cache] HIT for key: {key[:40]}...")
            return value
        else:
            _cache_misses += 1
            logger.info(f"[Cache] MISS for key: {key[:40]}...")
            return None
    except Exception as e:
        _cache_misses += 1
        logger.warning(f"[Cache] Redis GET failed ({e}). Treating as cache miss.")
        return None


def cache_set(query: str, response: str, ttl: int = None) -> bool:
    """
    Store a response in Redis for the given query.

    Args:
        query:    The user query (will be normalized for key generation)
        response: The final response string to cache
        ttl:      TTL in seconds (defaults to CACHE_TTL_SECONDS)

    Returns:
        True if stored successfully, False otherwise.
    """
    client = _get_redis_client()
    if client is None:
        return False
    if not response or not response.strip():
        return False
    key = generate_cache_key(query)
    effective_ttl = ttl if ttl is not None else CACHE_TTL_SECONDS
    try:
        client.setex(key, effective_ttl, response)
        logger.info(f"[Cache] SET key: {key[:40]}... (TTL={effective_ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"[Cache] Redis SET failed ({e}). Response not cached.")
        return False


def cache_delete(query: str) -> bool:
    """Delete a cached response. Mainly for testing."""
    client = _get_redis_client()
    if client is None:
        return False
    key = generate_cache_key(query)
    try:
        client.delete(key)
        return True
    except Exception:
        return False


def reset_redis_client():
    """
    Reset the Redis client singleton.
    Used in tests to simulate Redis unavailability or reconnection.
    """
    global _redis_client, _redis_failed
    _redis_client = None
    _redis_failed = False

