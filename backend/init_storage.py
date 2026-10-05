"""Idempotent local S3 bucket initialization."""
from botocore.exceptions import ClientError
from .storage import configured_storage
from indicmeet.settings import get_settings
def main():
    store=configured_storage(get_settings().media_dir)
    try:store.client.head_bucket(Bucket=store.bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404","NoSuchBucket","NotFound"}:raise
        store.client.create_bucket(Bucket=store.bucket)
    print("STORAGE INIT ready")
if __name__=="__main__":main()
