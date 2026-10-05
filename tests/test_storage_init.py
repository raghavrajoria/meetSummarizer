
from types import SimpleNamespace
from botocore.exceptions import ClientError
from backend import init_storage
def test_bucket_init_creates_missing_and_is_safe_when_present(monkeypatch):
    calls=[]
    class Client:
        exists=False
        def head_bucket(self,**kwargs):
            if not self.exists:raise ClientError({"Error":{"Code":"404"}},"HeadBucket")
        def create_bucket(self,**kwargs):calls.append(kwargs);self.exists=True
    client=Client()
    monkeypatch.setattr(init_storage,"configured_storage",lambda root:SimpleNamespace(client=client,bucket="test-bucket"))
    init_storage.main();init_storage.main()
    assert calls==[{"Bucket":"test-bucket"}]
