"""Production refuses unsafe defaults before connecting to infrastructure."""
import json,os,subprocess,sys
import pytest
from backend.config_guard import validate_app_configuration
from backend.security import password_hash

@pytest.fixture
def production(monkeypatch):
    values={'APP_ENV':'production','DEMO':'false','DEMO_MODE':'false','ASR_MODE':'remote','ASR_SERVICE_URL':'https://asr.internal','ASR_SERVICE_TOKEN':'Asr-private-Q8!bC91xT6','GROQ_API_KEY':'Groq-private-Q7!aW82cF5','STORAGE_BACKEND':'s3','DATABASE_URL':'postgresql://user:Db-Q1!cV8aR6tS4xB2@db/meetings','REDIS_URL':'rediss://queue:6379/0','S3_ENDPOINT_URL':'https://storage.internal','S3_ACCESS_KEY':'Storage-Q9!bY7kE3vZ8','S3_SECRET_KEY':'Storage-Q2!dR6jB8xU4','MEDIA_SIGNING_SECRET':'Signing-Q8!zW4aR7xB6tC2vY9jU3pL5nF1','CORS_ORIGINS':'https://meetings.internal','AUTH_USERS_JSON':json.dumps({'owner':{'password_hash':password_hash('Random-Q2!cY7nB5sD8')}}),'API_TOKENS':''}
    for key,value in values.items():monkeypatch.setenv(key,value)
    for name in ('DEBUG','APP_DEBUG','FAKE_ASR','FAKE_LLM'):monkeypatch.delenv(name,raising=False)
    return values

def test_safe_production_config_and_default_remote(production):validate_app_configuration()

@pytest.mark.parametrize('name,value,message',[('DEMO','true','DEMO=true'),('DEMO_MODE','true','obsolete'),('ASR_MODE','import','ASR_MODE'),('MEDIA_SIGNING_SECRET','local-demo-signing-secret-change-before-hosting','MEDIA_SIGNING_SECRET'),('S3_SECRET_KEY','minioadmin','S3_SECRET_KEY'),('CORS_ORIGINS','*','wildcard'),('DEBUG','true','debug'),('AUTH_USERS_JSON',json.dumps({'demo':{'password_hash':'bad'}}),'AUTH_USERS_JSON')])
def test_production_refuses_demo_or_weak_values(production,monkeypatch,name,value,message):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError,match=message):validate_app_configuration()

def test_entrypoint_refuses_demo_before_network(production,monkeypatch):
    monkeypatch.setenv('DEMO','true')
    result=subprocess.run([sys.executable,'-m','backend.entrypoint'],capture_output=True,text=True,timeout=5)
    assert result.returncode!=0 and 'Production refuses DEMO=true' in result.stderr

def test_demo_requires_explicit_switch(production,monkeypatch):
    monkeypatch.setenv('APP_ENV','demo');monkeypatch.setenv('DEMO','false')
    with pytest.raises(ValueError,match='explicit'):validate_app_configuration()
    monkeypatch.setenv('DEMO','true');validate_app_configuration()
