"""Scan the LOCAL the-stack-smol snapshot (all 30 languages, JSONL data.json) with
ScanCode licensedcode, in parallel, to build a license-diverse real-corpus eval
set. Each high-confidence match becomes a query {text, license(=ground truth),
lang, score}. Round-robins a capped number of files per language so every
language (not just Apache/MIT-heavy python) contributes -> better license-family
diversity. Writes cache/real_corpus_eval.json (keeps existing rows).

  python scan_stack_local.py --per-lang 1500 --workers 16
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SNAP = Path("/home/asyin/.cache/huggingface/hub/datasets--bigcode--the-stack-smol/"
            "snapshots/4a6938ce94446f324c6629e7de00ac591710044b/data")
CACHE = Path("cache/real_corpus_eval.json")
SCORE_THRESHOLD = 50
MIN_MATCH_CHARS = 80

_IDX = None


def _init():
    global _IDX
    from licensedcode.cache import get_index
    _IDX = get_index()


def _scan(task):
    """task = (lang, content) -> (lang, {text,license,score}) or None."""
    lang, content = task
    try:
        matches = [m for m in _IDX.match(query_string=content)
                   if m.score() >= SCORE_THRESHOLD and len(m.matched_text()) >= MIN_MATCH_CHARS]
    except Exception:
        return None
    if not matches:
        return None
    m = max(matches, key=lambda x: x.score())
    return lang, {"text": m.matched_text(), "label": 1, "license": m.rule.license_expression,
                  "score": m.score(), "lang": lang, "source": "the-stack-smol"}


def iter_tasks(per_lang):
    """Yield (lang, content) for up to per_lang files from each language's JSONL."""
    for d in sorted(SNAP.iterdir()):
        f = d / "data.json"
        if not f.exists():
            continue
        n = 0
        with f.open() as fh:
            for line in fh:
                if n >= per_lang:
                    break
                try:
                    content = json.loads(line).get("content", "") or ""
                except Exception:
                    continue
                if len(content) < 100:
                    continue
                n += 1
                yield (d.name, content)


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    corpus = json.loads(CACHE.read_text()) if CACHE.exists() else []
    # keep prior negatives; replace positives with the fresh diverse scan
    negs = [r for r in corpus if r.get("label") == 0]
    pos = []

    tasks = list(iter_tasks(args.per_lang))
    print(f"scanning {len(tasks)} files across languages with {args.workers} workers…", flush=True)

    per_lang_pos = Counter()
    done = 0
    with Pool(args.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(_scan, tasks, chunksize=8):
            done += 1
            if done % 2000 == 0:
                print(f"  scanned {done}/{len(tasks)}  positives={len(pos)}", flush=True)
            if res is not None:
                lang, row = res
                pos.append(row)
                per_lang_pos[lang] += 1

    CACHE.write_text(json.dumps(negs + pos, indent=2))
    lic = Counter(r["license"] for r in pos)
    print(f"\npositives: {len(pos)}  distinct license-expressions: {len(lic)}", flush=True)
    print("--- positives per language ---", flush=True)
    for l, n in per_lang_pos.most_common():
        print(f"  {n:5d}  {l}", flush=True)
    print("--- top 40 license expressions ---", flush=True)
    for k, n in lic.most_common(40):
        print(f"  {n:5d}  {k}", flush=True)
    print(f"\nwrote {CACHE} ({len(negs)} negs + {len(pos)} pos)", flush=True)


if __name__ == "__main__":
    main()
