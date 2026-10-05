"""Full offline demo user flow against an already running API."""
import argparse,json,time
from pathlib import Path
import requests
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--url",default="http://127.0.0.1:8000");args=parser.parse_args()
    base=args.url.rstrip("/");client=requests.Session()
    def request(method,path,**kwargs):
        reply=client.request(method,base+path,timeout=30,**kwargs);reply.raise_for_status();return reply
    assert request("GET","/config").json()["demo_mode"],"Smoke requires explicit demo mode"
    assert requests.get(base+"/meetings",timeout=5).status_code==401
    login=request("POST","/auth/login",json={"username":"demo","password":"demo-password"}).json()
    client.headers["Authorization"]="Bearer "+login["access_token"]
    clip=Path(__file__).resolve().parents[1]/"fixtures/demo.mp4"
    with clip.open("rb") as stream:
        meeting=request("POST","/meetings",data={"title":"Smoke demo","date":"2026-10-05"},files={"recording":("demo.mp4",stream,"video/mp4")}).json()
    mid=meeting["id"];jobid=meeting["job"]["id"]
    deadline=time.monotonic()+120
    while True:
        job=request("GET",f"/jobs/{jobid}").json()
        print(f"job stage={job['stage']} progress={job['progress']} status={job['status']}")
        assert job["status"]!="failed",job
        if job["status"]=="done":break
        assert time.monotonic()<deadline,"Processing timeout";time.sleep(.5)
    result=request("GET",f"/meetings/{mid}").json()
    rows=result["transcript"];ids={s["segment_id"] for s in rows};assert rows
    claims=result["intelligence"]["overviewClaims"];assert claims
    assert all(c["source_segment_ids"] and set(c["source_segment_ids"])<=ids for c in claims)
    print(f"transcript segments={len(rows)} cited_overview_claims={len(claims)}")
    signed=request("GET",f"/meetings/{mid}/media-url").json()["url"]
    media=requests.get(base+signed,headers={"Range":"bytes=0-31"},timeout=10)
    assert media.status_code==206 and len(media.content)==32
    request("PUT",f"/meetings/{mid}/summary",json={"text":"Persisted smoke edit"})
    assert request("GET",f"/meetings/{mid}").json()["summary"]=="Persisted smoke edit"
    request("DELETE",f"/meetings/{mid}/summary")
    exported=request("GET",f"/meetings/{mid}/transcript")
    assert exported.json()==rows and "attachment" in exported.headers["Content-Disposition"]
    request("DELETE",f"/meetings/{mid}")
    assert client.get(base+f"/meetings/{mid}",timeout=5).status_code==404
    request("POST","/auth/logout")
    assert client.get(base+"/meetings",timeout=5).status_code==401
    print("SMOKE PASS login upload worker citations signed-range edit revert export delete logout")
if __name__=="__main__":main()
