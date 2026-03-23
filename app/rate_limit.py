import time
import uuid
import os
import streamlit as st
from supabase import create_client, Client

# Initialize Supabase client lazily
@st.cache_resource
def get_supabase_client() -> Client | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_KEY")
    if supa_url and supa_key:
        return create_client(supa_url, supa_key)
    return None

@st.cache_resource
def get_rate_limiter():
    """Returns a global dictionary to store timestamps per IP/session as fallback."""
    return {}

def check_rate_limit(limit=4, window=60):
    """
    Checks if the current user has exceeded the rate limit.
    Attempts to use Supabase 'rate_limits' table. Falls back to in-memory dict.
    Returns True if allowed, False if blocked.
    """
    ip = "unknown"
    try:
        # st.context requires Streamlit >= 1.37
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For") or headers.get("Remote-Addr", "unknown")
    except Exception:
        pass
        
    if ip == "unknown":
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        ip = st.session_state.session_id

    # Try Supabase first
    supabase = get_supabase_client()
    if supabase:
        try:
            from datetime import datetime, timedelta, timezone
            
            # Use UTC time for timestamps
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=window)
            cutoff_iso = cutoff.isoformat()
            
            # Prune old records for this IP to keep table clean
            supabase.table("rate_limits").delete().eq("ip_or_session", ip).lt("timestamp", cutoff_iso).execute()
            
            # Check how many requests have been made within the window
            result = supabase.table("rate_limits").select("id", count="exact").eq("ip_or_session", ip).gte("timestamp", cutoff_iso).execute()
            count = result.count if result.count is not None else 0
            
            if count >= limit:
                return False
                
            # If allowed, insert new record
            supabase.table("rate_limits").insert({"ip_or_session": ip}).execute()
            return True
        except Exception as e:
            # If table doesn't exist or DB error, print warning and fallback
            print(f"Warning: Supabase rate limiting failed (fallback to memory): {e}")
            pass

    # Fallback: In-memory dictionary
    limiter = get_rate_limiter()
    now_ts = time.time()
    
    if ip not in limiter:
        limiter[ip] = []
        
    # Remove timestamps older than window
    limiter[ip] = [t for t in limiter[ip] if now_ts - t < window]
    
    if len(limiter[ip]) >= limit:
        return False
        
    limiter[ip].append(now_ts)
    return True
