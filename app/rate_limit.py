import time
import uuid
import os
import streamlit as st

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_redis_client():
    """Returns a Redis client if REDIS_URL is configured, else None."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()  # fail fast if unreachable
        return client
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")
        return None


@st.cache_resource
def get_supabase_client():
    """Returns a Supabase client if credentials are configured, else None."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_KEY")
    if supa_url and supa_key:
        try:
            from supabase import create_client
            return create_client(supa_url, supa_key)
        except Exception as e:
            print(f"Warning: Supabase client init failed: {e}")
    return None


@st.cache_resource
def _get_memory_store():
    """In-process fallback store: {ip -> [timestamp, ...]}"""
    return {}


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _check_redis(client, ip: str, limit: int, window: int) -> bool:
    """
    Sliding-window rate limit via a Redis sorted set.
    Fully atomic via pipeline — safe under concurrent Streamlit workers.
    Returns True if the request is allowed.
    """
    key = f"rate_limit:{ip}"
    now = time.time()

    pipe = client.pipeline()
    pipe.zremrangebyscore(key, 0, now - window)   # prune expired entries
    pipe.zadd(key, {f"{now}:{uuid.uuid4()}": now}) # add unique member
    pipe.zcard(key)                                # count within window
    pipe.expire(key, window)                       # auto-TTL cleanup
    _, _, count, _ = pipe.execute()

    if count > limit:
        # Undo the entry we just added — we're over the limit
        # (Best-effort; slight over-count is harmless for rate limiting.)
        client.zremrangebyscore(key, now, now + 0.001)
        return False
    return True


def _check_supabase(client, ip: str, limit: int, window: int) -> bool:
    """
    Sliding-window rate limit via Supabase 'rate_limits' table.
    Returns True if the request is allowed.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    cutoff_iso = (now - timedelta(seconds=window)).isoformat()

    # Prune stale records
    client.table("rate_limits").delete().eq("ip_or_session", ip).lt("timestamp", cutoff_iso).execute()

    # Count recent requests
    result = (
        client.table("rate_limits")
        .select("id", count="exact")
        .eq("ip_or_session", ip)
        .gte("timestamp", cutoff_iso)
        .execute()
    )
    count = result.count if result.count is not None else 0

    if count >= limit:
        return False

    client.table("rate_limits").insert({"ip_or_session": ip}).execute()
    return True


def _check_memory(ip: str, limit: int, window: int) -> bool:
    """Simple in-memory sliding-window fallback."""
    store = _get_memory_store()
    now = time.time()

    timestamps = store.get(ip, [])
    timestamps = [t for t in timestamps if now - t < window]

    if len(timestamps) >= limit:
        store[ip] = timestamps
        return False

    timestamps.append(now)
    store[ip] = timestamps
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_identifier() -> str:
    """Return the best available identifier: IP header → session UUID."""
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For") or headers.get("Remote-Addr", "")
        if ip:
            return ip
    except Exception:
        pass

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def check_rate_limit(limit: int = 4, window: int = 60) -> bool:
    """
    Check whether the current user is within the rate limit.

    Priority: Redis → Supabase → in-memory.
    Returns True if the request should be allowed, False if blocked.
    """
    ip = _resolve_identifier()

    # 1. Redis (preferred)
    redis_client = get_redis_client()
    if redis_client:
        try:
            return _check_redis(redis_client, ip, limit, window)
        except Exception as e:
            print(f"Warning: Redis rate-limit check failed (falling back): {e}")

    # 2. Supabase
    supabase_client = get_supabase_client()
    if supabase_client:
        try:
            return _check_supabase(supabase_client, ip, limit, window)
        except Exception as e:
            print(f"Warning: Supabase rate-limit check failed (falling back): {e}")

    # 3. In-memory (last resort)
    return _check_memory(ip, limit, window)
