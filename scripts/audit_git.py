"""Read every reachable git blob, including document XML, without printing secrets."""
import io,re,subprocess,zipfile
def git(*args):return subprocess.check_output(["git",*args])
def main():
    patterns=[rb"gsk_[A-Za-z0-9]{20,}",rb"hf_[A-Za-z0-9]{20,}",rb"(?:AKIA|ASIA)[A-Z0-9]{16}",rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    objects=git("rev-list","--objects","--all").decode().splitlines()
    blobs,large,hits=0,[],[]
    paths={entry.partition(" ")[0]:entry.partition(" ")[2] for entry in objects}
    stream=io.BytesIO(subprocess.check_output(["git","cat-file","--batch"],input=("\n".join(paths)+"\n").encode()))
    for entry in objects:
        header=stream.readline().decode().strip().split()
        oid,kind,size=header;data=stream.read(int(size));stream.read(1)
        if kind!="blob":continue
        path=paths[oid];blobs+=1
        if len(data)>5*1024*1024:large.append((path,len(data)))
        contents=[data]
        if data.startswith(b"PK"):
            try:
                archive=zipfile.ZipFile(io.BytesIO(data))
                contents += [archive.read(name) for name in archive.namelist() if name.endswith(".xml")]
            except zipfile.BadZipFile:pass
        if any(re.search(pattern,content) for pattern in patterns for content in contents):hits.append((oid,path))
    tracked=git("ls-files").decode().splitlines()
    suspicious=[p for p in tracked if re.search(r"(^|/)(?:\.env(?:\.|$)|node_modules/|\.venv/|__pycache__/|output/)|\.(?:safetensors|ckpt|pt|pth|onnx)$",p) and p not in {".env.example",".env.demo.example"}]
    print(f"HISTORY blobs={blobs} credential_pattern_hits={len(hits)} blobs_over_5MB={len(large)}")
    print(f"TRACKED suspicious_files={len(suspicious)}")
    for oid,path in hits:print("CREDENTIAL MATCH (value redacted)",oid,path)
    for path,size in large:print("LARGE",path,size)
    for path in suspicious:print("SUSPICIOUS",path)
    sizes=[]
    for entry in git("ls-tree","-rl","HEAD").decode().splitlines():
        metadata,path=entry.split("\t",1);sizes.append((int(metadata.split()[-1]),path))
    for size,path in sorted(sizes,reverse=True)[:10]:print("LARGEST TRACKED",size,path)
    for needle in ("gsk_","hf_","AKIA","ASIA","PRIVATE KEY"):
        history=git("log","--all","--oneline","-S",needle).decode().strip()
        print(f"git log --all --oneline -S {needle}:\n{history or '(none)'}")
    return int(bool(hits or large or suspicious))
if __name__=="__main__":raise SystemExit(main())
