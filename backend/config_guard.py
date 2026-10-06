"""Fail closed at application startup; errors never echo secret values."""
import json,os
from urllib.parse import urlparse
from indicmeet.settings import get_settings

def unsafe(value):
    low=value.lower()
    return any(word in low for word in ('demo','change-me','changeme','replace-me','default','minioadmin','password','your-secret','example-secret','test-signing','local-proof')) or len(value)<16 or len(set(value))<8

def allowed_transport(value, local_real=False):
    parsed=urlparse(value)
    if parsed.username or parsed.password or not parsed.hostname:return False
    return parsed.scheme=='https' or (local_real and parsed.scheme=='http' and parsed.hostname in {'127.0.0.1','localhost','host.docker.internal'})

def validate_app_configuration():
    settings=get_settings();errors=[]
    profile=os.environ.get('DEPLOY_PROFILE','').strip()
    local_real=profile=='local-real'
    if profile not in {'','local-real'}:errors.append('Unknown DEPLOY_PROFILE')
    if local_real and (settings.app_env!='production' or settings.demo_mode):errors.append('local-real requires production and refuses demo')
    if settings.app_env not in {'production','development','demo'}:errors.append('APP_ENV must be production, development or demo')
    if os.environ.get('DEMO_MODE','false').lower()=='true':errors.append('DEMO_MODE is obsolete; demo requires explicit DEMO=true')
    if settings.app_env=='production' and settings.demo_mode:errors.append('Production refuses DEMO=true')
    if settings.app_env=='demo' and not settings.demo_mode:errors.append('Demo requires explicit DEMO=true')
    if any(os.environ.get(k,'false').lower() in {'true','1','yes'} for k in ('DEBUG','APP_DEBUG','FAKE_ASR','FAKE_LLM')) and not settings.demo_mode:errors.append('Production refuses debug/fake flags')
    if '*' in settings.cors_origins:errors.append('CORS wildcard is forbidden')
    if settings.demo_mode and settings.app_env!='production' and not errors:return
    if settings.app_env=='production':
        if settings.asr_mode!='remote':errors.append('Production requires ASR_MODE=remote')
        if not settings.asr_service_url or not allowed_transport(settings.asr_service_url,local_real):errors.append('Production requires HTTPS ASR_SERVICE_URL base URL (local-real HTTP is loopback only)')
        if not settings.asr_service_token or unsafe(settings.asr_service_token):errors.append('Production requires private ASR_SERVICE_TOKEN')
        if not settings.groq_api_key or unsafe(settings.groq_api_key):errors.append('Production requires private GROQ_API_KEY')
        if os.environ.get('STORAGE_BACKEND')!='s3':errors.append('Production requires external S3 storage')
        if not settings.database_url.startswith(('postgresql://','postgresql+psycopg://')):errors.append('Production requires PostgreSQL DATABASE_URL')
        dbpassword=urlparse(settings.database_url).password or ''
        if unsafe(dbpassword):errors.append('DATABASE_URL password is missing/default/weak')
        for name in ('S3_ACCESS_KEY','S3_SECRET_KEY'):
            if unsafe(os.environ.get(name,'')):errors.append(name+' is missing/default/weak')
        if os.environ.get('S3_ENDPOINT_URL') and not allowed_transport(os.environ['S3_ENDPOINT_URL'],local_real):errors.append('Production requires HTTPS S3_ENDPOINT_URL (local-real HTTP is loopback only)')
        if not os.environ.get('REDIS_URL'):errors.append('Production requires REDIS_URL')
        if not settings.cors_origins or any(not allowed_transport(origin,local_real) or urlparse(origin).path not in {'','/'} or urlparse(origin).query or urlparse(origin).fragment for origin in settings.cors_origins):errors.append('Production requires exact HTTPS CORS_ORIGINS (local-real HTTP is loopback only)')
        secret=os.environ.get('MEDIA_SIGNING_SECRET','')
        if len(secret)<32 or unsafe(secret):errors.append('MEDIA_SIGNING_SECRET is missing/default/weak')
        try:
            users=json.loads(os.environ.get('AUTH_USERS_JSON') or '{}')
            if not isinstance(users,dict) or not users:raise ValueError()
            from .security import password_matches
            for name,user in users.items():
                stored=user.get('password_hash','')
                parts=stored.split('$')
                if len(parts)!=4 or len(parts[2])<16 or len(parts[3])!=64:raise ValueError()
                int(parts[3],16)
                if user.get('role','editor') not in {'viewer','editor','admin'}:raise ValueError()
                if name.lower()=='demo' or not stored.startswith('pbkdf2_sha256$310000$') or any(password_matches(p,stored) for p in ('demo-password','password','changeme','minioadmin')):raise ValueError()
        except (ValueError,TypeError,AttributeError):errors.append('AUTH_USERS_JSON requires hashed non-demo users')
        if any(unsafe(value) for value in settings.api_tokens):errors.append('API_TOKENS contains a default/weak token')
    if errors:raise ValueError('Unsafe application configuration: '+'; '.join(errors))
