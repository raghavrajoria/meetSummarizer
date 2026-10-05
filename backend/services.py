"""Redis wakeups; durable job ownership always remains in the database."""
import logging,os
def redis_client():
    if not os.environ.get("REDIS_URL"):return None
    import redis
    return redis.Redis.from_url(os.environ["REDIS_URL"],socket_connect_timeout=2,socket_timeout=5)
def notify():
    client=redis_client()
    if client:
        try:client.lpush("indicmeet:wakeup","1");client.ltrim("indicmeet:wakeup",0,999)
        except Exception:logging.getLogger("indicmeet.queue").warning("queue_wakeup_unavailable fallback=database_poll")
def wait(seconds):
    client=redis_client()
    if client:
        try:client.blpop("indicmeet:wakeup",timeout=max(1,int(seconds)));return
        except Exception:pass
    __import__("time").sleep(max(.1,seconds))
def dependencies_ready(store):
    if not store.ready():return False
    client=redis_client()
    return bool(client.ping()) if client else True
