"""Real ASR v2 client acceptance; use only authorized teammate/local endpoints."""
import argparse,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from indicmeet.remote_asr import RemoteAsrProvider
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--audio',type=pathlib.Path,required=True);args=parser.parse_args()
    provider=RemoteAsrProvider(args.url);health=provider.health();rows=provider.transcribe(args.audio)
    assert len({s['segment_id'] for s in rows})==len(rows)
    assert all(s['asr']['model_version']==health['model_version'] for s in rows)
    print('ASR V2 SMOKE PASS segments='+str(len(rows))+' model_version='+health['model_version'])
if __name__=='__main__':main()
