"""Real-corpus eval for the Nirjas gate using the-stack-smol + ScanCode.

Labels:
  positive — text that ScanCode confidently matched as license text (score >= 50)
  negative — header region of files where ScanCode found nothing

This tests what the gate actually does in production: classify sliding-window
chunks of real source files, not synthetic examples.

Usage:
    .venv/bin/python eval_real_corpus.py               # default 300+300
    .venv/bin/python eval_real_corpus.py --pos 500 --neg 500
    .venv/bin/python eval_real_corpus.py --no-cache    # force re-scan

Results saved to output/real_corpus_eval.json.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "cache" / "real_corpus_eval.json"
RESULTS_PATH = ROOT / "output" / "real_corpus_eval.json"
SCORE_THRESHOLD = 50   # licensedcode match confidence; 80+ = verbatim, 50+ = near-match
NEG_HEADER_CHARS = 1500  # how much of the file header to use as a negative sample

LANGUAGES = ["python", "javascript", "java", "go", "c", "cpp", "rust", "shell", "typescript", "ruby"]


def _read_hf_token() -> str | None:
    env = Path(".env")
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("HF_API_TOKEN="):
            return line.split("=", 1)[1].strip() or None
    return None


def scan_file(idx, content: str) -> list[dict]:
    """Return high-confidence license matches from licensedcode."""
    try:
        matches = idx.match(query_string=content)
        return [
            {"text": m.matched_text(), "license": m.rule.license_expression,
             "score": m.score(), "start_line": m.start_line, "end_line": m.end_line}
            for m in matches if m.score() >= SCORE_THRESHOLD and len(m.matched_text()) >= 80
        ]
    except Exception:
        return []


def stream_labeled(target_pos: int, target_neg: int, have_pos: int = 0, have_neg: int = 0) -> Iterator[dict]:
    """Stream labeled samples from the-stack-smol, skipping already-cached counts."""
    from datasets import load_dataset
    from licensedcode.cache import get_index

    print("Building licensedcode index (first run: ~10s)…")
    idx = get_index()
    print("Index ready.")

    token = _read_hf_token()
    pos, neg = have_pos, have_neg

    for lang in LANGUAGES:
        if pos >= target_pos and neg >= target_neg:
            break
        try:
            ds = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
                token=token,
            )
        except Exception as e:
            print(f"  skip {lang}: {e}")
            continue

        for row in ds:
            if pos >= target_pos and neg >= target_neg:
                break
            content: str = row.get("content", "") or ""
            if len(content) < 100:
                continue

            matches = scan_file(idx, content)

            if matches and pos < target_pos:
                # Use the highest-scoring match as the positive sample
                best = max(matches, key=lambda m: m["score"])
                yield {"text": best["text"], "label": 1, "license": best["license"],
                       "score": best["score"], "lang": lang, "source": "the-stack-smol"}
                pos += 1

            elif not matches and neg < target_neg:
                header = content[:NEG_HEADER_CHARS].strip()
                if len(header) < 50:
                    continue
                yield {"text": header, "label": 0, "license": None,
                       "score": 0.0, "lang": lang, "source": "the-stack-smol"}
                neg += 1

        print(f"  {lang}: +{pos} pos / +{neg} neg so far")


def build_or_load_corpus(target_pos: int, target_neg: int, force: bool) -> list[dict]:
    corpus: list[dict] = []
    if not force and CACHE_PATH.exists():
        corpus = json.loads(CACHE_PATH.read_text())

    cached_pos = sum(1 for r in corpus if r["label"] == 1)
    cached_neg = sum(1 for r in corpus if r["label"] == 0)

    if cached_pos >= target_pos and cached_neg >= target_neg:
        print(f"Loaded from cache: {cached_pos} pos, {cached_neg} neg")
        return [r for r in corpus if r["label"] == 1][:target_pos] + \
               [r for r in corpus if r["label"] == 0][:target_neg]

    need_pos = max(0, target_pos - cached_pos)
    need_neg = max(0, target_neg - cached_neg)
    print(f"Cache has {cached_pos} pos, {cached_neg} neg — streaming {need_pos} more pos, {need_neg} more neg…")
    new_samples = list(stream_labeled(target_pos, target_neg, cached_pos, cached_neg))
    corpus.extend(new_samples)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"Cached {len(corpus)} total samples → {CACHE_PATH}")
    return corpus


def run_eval(corpus: list[dict]) -> None:
    from train_nirjas_gate import load_gate, classify

    pipe, threshold = load_gate()
    print(f"Gate loaded  threshold={threshold}")

    texts = [r["text"] for r in corpus]
    preds = classify(pipe, texts, threshold)

    for r, p in zip(corpus, preds):
        r["predicted"] = int(p)
        r["correct"] = int(p) == r["label"]

    pos = [r for r in corpus if r["label"] == 1]
    neg = [r for r in corpus if r["label"] == 0]
    recall = sum(r["predicted"] for r in pos) / len(pos) if pos else float("nan")
    fpr = sum(r["predicted"] for r in neg) / len(neg) if neg else float("nan")
    fn = [r for r in pos if not r["predicted"]]
    fp = [r for r in neg if r["predicted"]]

    print(f"\n=== Real-corpus eval (the-stack-smol + ScanCode) ===")
    print(f"  Positives : {len(pos):4d}   Recall : {recall:.4f}")
    print(f"  Negatives : {len(neg):4d}   FPR    : {fpr:.4f}")
    print(f"  False negatives: {len(fn)}")
    if fn:
        print("  FN samples (first 8):")
        for r in fn[:8]:
            print(f"    score={r['score']:.0f}  [{r['license']}]  {r['text'][:80]!r}")
    if fp:
        print(f"  False positives: {len(fp)}")
        print("  FP samples (first 5):")
        for r in fp[:5]:
            print(f"    [{r['lang']}]  {r['text'][:80]!r}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"\nFull results → {RESULTS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=300)
    ap.add_argument("--neg", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    corpus = build_or_load_corpus(args.pos, args.neg, args.no_cache)
    pos = sum(1 for r in corpus if r["label"] == 1)
    neg = sum(1 for r in corpus if r["label"] == 0)
    print(f"Corpus: {pos} pos, {neg} neg")
    run_eval(corpus)


if __name__ == "__main__":
    main()
