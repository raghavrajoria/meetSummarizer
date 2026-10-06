"""Create private secrets without bypassing production configuration checks."""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_env(path):
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            name, value = line.split('=', 1)
            values[name.strip()] = value.strip().strip("'\"")
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.env.local-real'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Refusing to replace private configuration')
    db_password=secrets.token_urlsafe(32)
    redis_password=secrets.token_urlsafe(32)
    values={'DEPLOY_PROFILE':'local-real','DATABASE_URL':f'postgresql+psycopg://indicmeet:{db_password}@127.0.0.1:55432/indicmeet','POSTGRES_PASSWORD':db_password,'REDIS_PASSWORD':redis_password,'REDIS_URL':f'redis://:{redis_password}@127.0.0.1:56379/0','STORAGE_BACKEND':'s3','S3_ENDPOINT_URL':'http://127.0.0.1:59000','S3_BUCKET':'indicmeet-local-real','S3_REGION':'us-east-1','S3_ACCESS_KEY':secrets.token_urlsafe(24),'S3_SECRET_KEY':secrets.token_urlsafe(32),'ASR_SERVICE_URL':'http://127.0.0.1:9002','CORS_ORIGINS':'http://127.0.0.1:5173,http://127.0.0.1:8080','INDICMEET_DATA_DIR':str(Path('data/local-real').resolve()),'ASR_STATE_DIR':str(Path('data/local-real/asr-state').resolve())}
    original=read_env(Path('.env'))
    from backend.security import password_hash
    password=secrets.token_urlsafe(32)
    values.update(APP_ENV='production',DEMO='false',DEMO_MODE='false',ASR_MODE='remote',ASR_DEVICE='cpu',WHISPER_MODEL='small',WHISPER_COMPUTE_TYPE='int8',ASR_RAM_HEADROOM_GB='3',ASR_CPU_THREADS='2',ASR_MODEL_VERSION='cpu-small-int8-indic600m-mms256-20261006',ASR_SERVICE_TOKEN=secrets.token_urlsafe(32),MEDIA_SIGNING_SECRET=secrets.token_urlsafe(48),AUTH_USERS_JSON=json.dumps({'localreal':{'password_hash':password_hash(password),'role':'admin'}}),LOCAL_REAL_USERNAME='localreal',LOCAL_REAL_PASSWORD=password,GROQ_API_KEY=original.get('GROQ_API_KEY',''),STRICT_REVIEW='true')
    for key in ('HF_TOKEN','HUGGINGFACE_TOKEN'):
        if original.get(key): values[key]=original[key]
    values.update(GROQ_MAX_LIVE_CALLS='24', GROQ_MAX_ATTEMPTS='2', INDICMEET_LLM_CACHE=str(Path('data/local-real/llm-cache').resolve()), ACCESS_TOKEN_SECONDS='3600', HF_HUB_DISABLE_XET='1')
    os.environ.update(values)
    from backend.config_guard import validate_app_configuration
    validate_app_configuration()
    with args.output.open('x',encoding='utf-8') as stream:
        for name,value in values.items():
            if '\n' in value or "'" in value: raise ValueError('Unsupported configuration quoting')
            stream.write(f"{name}='{value}'\n")
    os.chmod(args.output,0o600)
    print('Private local-real configuration generated; production safety validation passed. No secrets displayed.')


if __name__=='__main__': main()
