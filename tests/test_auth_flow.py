import json,time
from backend.security import password_hash
from backend.models import AccessSession
def login(env,monkeypatch,role="editor"):
    monkeypatch.setenv("AUTH_USERS_JSON",json.dumps({"tester":{"password_hash":password_hash("correct"),"role":role}}))
    reply=env.client.post("/auth/login",json={"username":"tester","password":"correct"})
    assert reply.status_code==200
    return reply.json()["access_token"]
def test_login_reject_expire_logout_and_roles(api_env,monkeypatch):
    c=api_env.client
    assert c.get("/meetings").status_code==401
    token=login(api_env,monkeypatch,"viewer")
    assert c.post("/auth/login",json={"username":"tester","password":"wrong"}).status_code==401
    c.headers["Authorization"]="Bearer "+token
    assert c.get("/meetings").status_code==200
    assert c.post("/meetings").status_code==403
    assert c.post("/auth/logout").status_code==204
    assert c.get("/meetings").status_code==401
    token=login(api_env,monkeypatch)
    from backend.security import token_hash
    with api_env.sessions() as db:
        row=db.get(AccessSession,token_hash(token)); row.expires=int(time.time())-1;db.commit()
    c.headers["Authorization"]="Bearer "+token
    assert c.get("/meetings").status_code==401
def test_media_signature_range_expiry_and_saved_edit(api_env,api_client,monkeypatch):
    c=api_client
    payload={"id":"auth-test","title":"Auth test","segments":[],"summary":"original"}
    assert c.post("/sessions/import",data={"session_json":json.dumps(payload)},files={"media":("clip.wav",b"0123456789","audio/wav")}).status_code==201
    token=login(api_env,monkeypatch); c.headers["Authorization"]="Bearer "+token
    url=c.get("/meetings/auth-test/media-url").json()["url"]
    c.headers.pop("Authorization")
    assert c.get("/meetings/auth-test/media").status_code==401
    response=c.get(url,headers={"Range":"bytes=2-4"})
    assert response.status_code==206 and response.content==b"234"
    assert c.get(url.replace("signature=","signature=bad")).status_code==401
    now = time.time()
    with monkeypatch.context() as clock:
        clock.setattr("backend.security.time.time",lambda:now+121)
        assert c.get(url).status_code==401
    c.headers["Authorization"]="Bearer test-token"
    assert c.put("/meetings/auth-test/summary",json={"text":"edited"}).status_code==200
    response=c.get("/meetings/auth-test").json()
    assert response["summary"]=="edited" and response["original_summary"]=="original"
    assert c.delete("/meetings/auth-test/summary").status_code==204
    assert c.get("/meetings/auth-test").json()["summary"]=="original"


def test_summary_edit_migration_preserves_original(api_env,api_client):
    from backend.models import MeetingSession,SummaryEdit
    with api_env.sessions() as db:
        db.add(MeetingSession(id="edits",title="Original",payload={"summary":"AI original","segments":[]}));db.commit()
    assert api_client.put("/meetings/edits/summary",json={"text":"human edit"}).status_code==200
    with api_env.sessions() as db:
        assert db.get(MeetingSession,"edits").payload["summary"]=="AI original"
        assert db.get(SummaryEdit,"edits").text=="human edit"


def test_compose_empty_users_value_allows_only_explicit_demo(api_env,monkeypatch):
    monkeypatch.setenv("AUTH_USERS_JSON", "")
    monkeypatch.setenv("DEMO_MODE", "true")
    response=api_env.client.post("/auth/login",json={"username":"demo","password":"demo-password"})
    assert response.status_code==200
    monkeypatch.setenv("DEMO_MODE", "false")
    assert api_env.client.post("/auth/login",json={"username":"demo","password":"demo-password"}).status_code==401
