"""Validate raw and adapted real/fixture transcripts without modifying their source."""
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from indicmeet.contract import Segment, canonicalize

def main():
    total, failed = 0, 0
    for target in sys.argv[1:] or ["fixtures"]:
        path = pathlib.Path(target)
        paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for source in paths:
            data = json.loads(source.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("segments", data.get("transcript", [])) if isinstance(data, dict) else []
            if not rows or not isinstance(rows[0], dict) or not any(k in rows[0] for k in ("text", "tx", "text_native")):
                continue
            before = 0
            for row in rows:
                try:
                    Segment.model_validate(row)
                    before += 1
                except ValueError:
                    pass
            try:
                adapted = canonicalize(rows)
                after = sum(bool(Segment.model_validate(row)) for row in adapted)
                print(f"{source}: before={before}/{len(rows)} after={after}/{len(rows)}")
                total += after
            except ValueError as exc:
                print(f"{source}: FAILED {type(exc).__name__}")
                failed += 1
    print(f"VALIDATED segments={total} failed_artifacts={failed}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
