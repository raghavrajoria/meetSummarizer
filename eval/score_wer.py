"""Score saved AGM bake-off outputs against the two specified reference windows."""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEADER = "reference provenance unverified (likely LLM/Gemini); 3 minutes of one recording; smoke test only"
WINDOWS = {"EN": (155.0, 221.0), "HI": (235.0, 347.0)}
SYSTEM_PREFIXES = ("A1_", "A2_", "B_", "C_", "D_")


@dataclass(frozen=True)
class WordCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int

    @property
    def wer(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / self.reference_words if self.reference_words else 0.0


def normalize(text: str) -> str:
    """Apply the required normalization identically to references and hypotheses."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Levenshtein counts, preferring substitution, deletion, then insertion on ties."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], op[i][0] = i, "D"
    for j in range(1, cols):
        cost[0][j], op[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            choices = (
                (cost[i - 1][j - 1] + (not same), "M" if same else "S", 0),
                (cost[i - 1][j] + 1, "D", 1),
                (cost[i][j - 1] + 1, "I", 2),
            )
            best = min(choices, key=lambda choice: (choice[0], choice[2]))
            cost[i][j], op[i][j] = int(best[0]), best[1]
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        step = op[i][j]
        if step == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif step == "D":
            deletions += 1
            i -= 1
        elif step == "I":
            insertions += 1
            j -= 1
        else:
            i -= 1
            j -= 1
    return substitutions, deletions, insertions


def score_words(reference: str, hypothesis: str) -> WordCounts:
    """Use Jiwer's alignment when installed, otherwise an equivalent local fallback."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    try:
        from jiwer import process_words
    except ImportError:
        subs, dels, ins = _edit_counts(ref_words, hyp_words)
    else:
        result = process_words(reference, hypothesis)
        subs, dels, ins = result.substitutions, result.deletions, result.insertions
    return WordCounts(subs, dels, ins, len(ref_words), len(hyp_words))


def _cer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import process_characters
    except ImportError:
        ref_chars, hyp_chars = list(reference), list(hypothesis)
        distance = sum(_edit_counts(ref_chars, hyp_chars))
        return distance / len(ref_chars) if ref_chars else (0.0 if not hyp_chars else 1.0)
    return process_characters(reference, hypothesis).cer


def _records(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next((payload[key] for key in ("segments", "asr", "transcript", "records") if isinstance(payload.get(key), list)), None)
    else:
        records = None
    if records is None or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{source}: expected an array or an object containing a segment array")
    return records


def _timestamp_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.strip().split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
    raise ValueError(f"invalid segment timestamp: {value!r}")


def _segment_interval(record: dict[str, Any]) -> tuple[float, float]:
    start_value = record.get("start", record.get("t", record.get("timestamp")))
    if start_value is None:
        raise ValueError("segment lacks start/t/timestamp")
    start = _timestamp_seconds(start_value)
    if record.get("end") is None:
        raise ValueError("segment lacks end; midpoint filtering requires explicit intervals")
    end = _timestamp_seconds(record["end"])
    if end < start:
        raise ValueError(f"segment end {end} precedes start {start}")
    return start, end


def _segment_text(record: dict[str, Any]) -> str:
    value = record.get("text", record.get("tx", record.get("en", "")))
    if not isinstance(value, str):
        raise ValueError("segment text must be a string")
    return value


def _reference_text(records: list[dict[str, Any]], window: str) -> str:
    # EN uses records timestamped 12:35 through 13:38; HI is the single 13:55 record.
    bounds = {"EN": (755.0, 818.0), "HI": (835.0, 835.0)}
    lower, upper = bounds[window]
    selected: list[tuple[float, str]] = []
    for record in records:
        timestamp = record.get("timestamp", record.get("start"))
        if timestamp is None:
            raise ValueError("reference record lacks timestamp/start")
        at = _timestamp_seconds(timestamp)
        if lower <= at <= upper:
            text = record.get("text", "")
            if not isinstance(text, str):
                raise ValueError("reference text must be a string")
            selected.append((at, text))
    return normalize(" ".join(text for _, text in sorted(selected, key=lambda item: item[0])))


def mostly_latin(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    latin = sum(unicodedata.name(char, "").startswith("LATIN") for char in letters)
    return latin / len(letters) > 0.5


def _load_systems(bakeoff: Path) -> list[tuple[str, Path]]:
    systems = []
    for folder in sorted(path for path in bakeoff.iterdir() if path.is_dir()):
        if folder.name.startswith(SYSTEM_PREFIXES):
            source = folder / "agm.json"
            if source.is_file():
                systems.append((folder.name, source))
    return systems


def _score_system(name: str, source: Path, reference_records: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    hypotheses = _records(json.loads(source.read_text(encoding="utf-8")), source)
    timed = []
    for record in hypotheses:
        start, end = _segment_interval(record)
        timed.append((start, end, record))
    clip_count = sum(1 for start, end, _ in timed if 0 <= (start + end) / 2 <= 600)
    skipped = clip_count < 3
    if skipped:
        print(f"SKIP {name}: only {clip_count} segment(s) in 0–600 s AGM clip (minimum is 3); hypothesis files will still be written")

    results = []
    for window, (start, end) in WINDOWS.items():
        selected = [
            ((seg_start + seg_end) / 2, _segment_text(record))
            for seg_start, seg_end, record in timed
            if start - 3 <= (seg_start + seg_end) / 2 <= end + 3
        ]
        hypothesis_raw = " ".join(text for _, text in sorted(selected, key=lambda item: item[0]))
        hypothesis = normalize(hypothesis_raw)
        reference = _reference_text(reference_records, window)
        (out_dir / f"{name}_{window}.txt").write_text(hypothesis_raw + "\n", encoding="utf-8")
        if skipped:
            continue
        words = score_words(reference, hypothesis)
        results.append({
            "system": name, "window": window, "wer": words.wer, "cer": _cer(reference, hypothesis),
            "reference_words": words.reference_words, "hypothesis_words": words.hypothesis_words,
            "substitutions": words.substitutions, "deletions": words.deletions,
            "insertions": words.insertions,
            "mostly_latin": window == "HI" and mostly_latin(hypothesis_raw),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.bakeoff.is_dir():
        parser.error(f"bake-off directory does not exist: {args.bakeoff}")
    if not args.ref.is_file():
        parser.error(f"reference file does not exist: {args.ref}")
    try:
        from jiwer import process_words  # noqa: F401
    except ImportError:
        print("jiwer is missing; using the built-in edit-distance fallback. No packages were installed.", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    reference_records = _records(json.loads(args.ref.read_text(encoding="utf-8")), args.ref)
    systems = _load_systems(args.bakeoff)
    if not systems:
        parser.error(f"no A1_*, A2_*, B_*, C_*, or D_* subfolders containing agm.json found in {args.bakeoff}")
    rows = [row for name, source in systems for row in _score_system(name, source, reference_records, args.out)]
    print(HEADER)
    print("System | Window | CER | WER | Ref words | Hyp words | S | D | I | Note")
    print("-------|--------|-----|-----|-----------|-----------|---|---|---|-----")
    for row in rows:
        note = "mostly Latin; WER meaningless" if row["mostly_latin"] else ""
        print(f"{row['system']} | {row['window']} | {row['cer']:.4f} | {row['wer']:.4f} | "
              f"{row['reference_words']} | {row['hypothesis_words']} | {row['substitutions']} | "
              f"{row['deletions']} | {row['insertions']} | {note}")
    if not rows:
        print("No systems were scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
