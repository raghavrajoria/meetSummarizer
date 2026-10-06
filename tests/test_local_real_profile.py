import pytest
from tests.test_production_config import production
from backend.config_guard import validate_app_configuration, allowed_transport


@pytest.fixture
def local_real(production,monkeypatch):
    monkeypatch.setenv('DEPLOY_PROFILE','local-real')
    for name,value in {'ASR_SERVICE_URL':'http://127.0.0.1:9002','S3_ENDPOINT_URL':'http://localhost:59000','CORS_ORIGINS':'http://host.docker.internal:8080'}.items(): monkeypatch.setenv(name,value)


def test_local_real_retains_production_checks(local_real): validate_app_configuration()


@pytest.mark.parametrize('name,value',[('ASR_SERVICE_URL','http://asr.example.com'),('ASR_SERVICE_URL','http://127.0.0.1.evil.com'),('ASR_SERVICE_URL','http://127.0.0.2'),('S3_ENDPOINT_URL','http://minio:9000'),('CORS_ORIGINS','http://192.168.1.2'),('CORS_ORIGINS','*'),('DEMO','true'),('APP_ENV','development'),('S3_ACCESS_KEY','minioadmin'),('S3_SECRET_KEY','minioadmin'),('ASR_SERVICE_TOKEN','weak'),('MEDIA_SIGNING_SECRET','default'),('DEBUG','true'),('FAKE_LLM','true'),('AUTH_USERS_JSON','{"demo":{"password_hash":"bad"}}')])
def test_local_real_refuses_unsafe(local_real,monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError): validate_app_configuration()


@pytest.mark.parametrize('url',['http://127.0.0.1','http://localhost','http://host.docker.internal'])
def test_http_only_with_explicit_profile(url):
    assert allowed_transport(url,True)
    assert not allowed_transport(url,False)


def test_invalid_profile(production,monkeypatch):
    monkeypatch.setenv('DEPLOY_PROFILE','typo')
    with pytest.raises(ValueError,match='DEPLOY_PROFILE'): validate_app_configuration()
