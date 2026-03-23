import time
import uuid
import streamlit as st

@st.cache_resource
def get_rate_limiter():
    """Returns a global dictionary to store timestamps per IP/session."""
    return {}

def check_rate_limit(limit=4, window=60):
    """
    Checks if the current user has exceeded the rate limit.
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

    limiter = get_rate_limiter()
    now = time.time()
    
    if ip not in limiter:
        limiter[ip] = []
        
    # Remove timestamps older than window
    limiter[ip] = [t for t in limiter[ip] if now - t < window]
    
    if len(limiter[ip]) >= limit:
        return False
        
    limiter[ip].append(now)
    return True
