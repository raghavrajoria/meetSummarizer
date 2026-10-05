"""GPU host/Kaggle entry point; never load the model on the owner workstation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    parser=argparse.ArgumentParser();parser.add_argument("audio",type=Path);parser.add_argument("csv",type=Path);args=parser.parse_args()
    if args.csv.exists(): raise SystemExit("Refusing to overwrite existing speaker turns")
    from indicmeet.pipeline import _gpu_diarize
    _gpu_diarize(args.audio,args.csv)
if __name__=="__main__": main()
