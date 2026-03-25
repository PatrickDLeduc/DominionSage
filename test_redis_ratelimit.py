"""
Quick test for Redis-backed rate limiting.
Run with: .venv\Scripts\python test_redis_ratelimit.py
"""
import os, time
from dotenv import load_dotenv
load_dotenv()

import redis
from app.rate_limit import _check_redis

LIMIT = 4
WINDOW = 60
TEST_IP = "test-user-123"

r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

# Clean slate
r.delete(f"rate_limit:{TEST_IP}")

print(f"Sending {LIMIT + 2} requests (limit={LIMIT}, window={WINDOW}s)\n")
for i in range(1, LIMIT + 3):
    allowed = _check_redis(r, TEST_IP, LIMIT, WINDOW)
    status = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
    print(f"  Request {i}: {status}")

# Cleanup
r.delete(f"rate_limit:{TEST_IP}")
print("\nDone. Key cleaned up.")
