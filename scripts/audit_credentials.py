"""Commit/file attribution for every reachable known credential-pattern match."""
import io,json,re,subprocess,zipfile
PATTERNS=[rb"gsk_[A-Za-z0-9]{20,}",rb"hf_[A-Za-z0-9]{20,}",rb"AKIA[0-9A-Z]{16}",rb"ASIA[0-9A-Z]{16}",rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
def git(*args):return subprocess.check_output(['git',*args])
def main():
    commits=git('rev-list','--all').decode().splitlines()
    entries=git('rev-list','--objects','--all').decode().splitlines()
    identifiers=[line.split(' ',1)[0] for line in entries]
    stream=io.BytesIO(subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(identifiers)+'\n').encode()))
    matched=set();blobs=0
    for _ in identifiers:
        oid,kind,size=stream.readline().decode().split();data=stream.read(int(size));stream.read(1)
        if kind!='blob':continue
        blobs+=1;contents=[data]
        if data.startswith(b'PK'):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:contents.extend(archive.read(name) for name in archive.namelist() if name.endswith('.xml'))
            except zipfile.BadZipFile:pass
        if any(re.search(pattern,content) for pattern in PATTERNS for content in contents):matched.add(oid)
    hits=[]
    if matched:
        for commit in commits:
            for entry in git('ls-tree','-rz',commit).split(b'\0'):
                if not entry:continue
                metadata,path=entry.split(b'\t',1);oid=metadata.decode().split()[2]
                if oid in matched:hits.append({'commit':commit,'file':path.decode(errors='replace'),'blob':oid,'credential':'REDACTED'})
    print(f'CREDENTIAL AUDIT reachable_commits={len(commits)} blobs={blobs} matched_blobs={len(matched)} commit_file_hits={len(hits)}')
    for hit in hits:print(json.dumps(hit))
    if hits:return 1 # Caller must STOP; do not print matching values.
    quoted=set()
    for needle in ('gsk_','hf_'):
        for commit in git('log','--all','--format=%H','-S',needle).decode().splitlines():
            path=''
            for line in git('show',commit,'--format=','--unified=0').decode(errors='replace').splitlines():
                if line.startswith('+++ b/'):path=line[6:]
                elif line.startswith('--- a/') and not path:path=line[6:]
                elif line[:1] in {'+','-'} and not line.startswith(('+++','---')) and needle in line:
                    item=(commit,path,line[1:])
                    if item not in quoted:quoted.add(item);print('PICKAXE NON-CREDENTIAL LINE '+json.dumps({'commit':commit[:7],'file':path,'line':line[1:]}))
    print('NO REAL CREDENTIAL PATTERN MATCHES; quoted pickaxe lines are regex patterns, metadata identifiers, or report references')
    return 0
if __name__=='__main__':raise SystemExit(main())
