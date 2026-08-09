"""Evaluate the Nirjas gate against FOSSology's nomos hand-labeled testdata.

The nomos testdata is the closest thing to ground truth we have: real files
the FOSSology scanner is expected to handle, labeled by the nomos team.

Usage:
    .venv/bin/python eval_nomos_real.py
    .venv/bin/python eval_nomos_real.py --clone-only   # just fetch testdata
    .venv/bin/python eval_nomos_real.py --head 100     # quick smoke test

Results saved to output/nomos_eval_results.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


FOSSOLOGY_REPO = "https://github.com/fossology/fossology.git"
TESTDATA_SUBDIR = "src/nomos/agent_tests/testdata"
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "nomos_testdata"
RESULTS_PATH = ROOT / "output" / "nomos_eval_results.json"
# License headers always live at the top; pure license files are short.
# 1500 chars covers both without feeding huge source files to the gate.
MAX_CHARS = 1500


def clone_testdata() -> Path:
    """Sparse-clone only the nomos testdata subtree."""
    testdata_path = CACHE_DIR / TESTDATA_SUBDIR
    if testdata_path.exists() and any(testdata_path.iterdir()):
        print(f"Testdata already at {testdata_path}")
        return testdata_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Sparse-cloning FOSSology nomos testdata (no full repo download)…")
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth=1",
         FOSSOLOGY_REPO, str(CACHE_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", TESTDATA_SUBDIR],
        cwd=CACHE_DIR, check=True,
    )
    print(f"Testdata ready at {testdata_path}")
    return testdata_path


def parse_manifest(testdata_path: Path) -> dict[str, list[str]]:
    """Parse LastGoodNomosTestfilesScan → {relative_path: [license, ...]}."""
    manifest = testdata_path / "LastGoodNomosTestfilesScan"
    if not manifest.exists():
        sys.exit(f"Manifest not found: {manifest}")

    labeled: dict[str, list[str]] = {}
    for line in manifest.read_text(errors="replace").splitlines():
        # Format: "File NomosTestfiles/MIT/MIT.txt contains license(s) MIT"
        if not line.startswith("File "):
            continue
        parts = line.split(" contains license(s) ")
        if len(parts) != 2:
            continue
        rel = parts[0].removeprefix("File ").strip()
        licenses = [l.strip() for l in parts[1].split(",") if l.strip()]
        labeled[rel] = licenses
    return labeled


def read_head(path: Path) -> str:
    try:
        return path.read_text(errors="replace")[:MAX_CHARS]
    except OSError:
        return ""


def run_eval(testdata_path: Path, labeled: dict[str, list[str]], head: int | None):
    from train_nirjas_gate import load_gate, classify

    pipe, threshold = load_gate()
    print(f"Gate loaded  threshold={threshold}")

    # Build evaluation set
    # Positives: all files in manifest (they contain a detected license)
    # Negatives: noLic, empty (explicitly no-license files not in manifest)
    neg_names = {"noLic", "empty"}
    records = []

    for rel, licenses in labeled.items():
        path = testdata_path / rel
        records.append({"path": rel, "label": 1, "licenses": licenses, "file": path})

    for name in neg_names:
        path = testdata_path / name
        if path.exists():
            records.append({"path": name, "label": 0, "licenses": [], "file": path})

    if head:
        records = records[:head]

    print(f"Evaluating {len(records)} files ({sum(r['label'] for r in records)} pos, "
          f"{sum(1 - r['label'] for r in records)} neg)…")

    texts = [read_head(r["file"]) for r in records]
    predictions = classify(pipe, texts, threshold)

    results = []
    for r, pred in zip(records, predictions):
        results.append({
            "path": r["path"],
            "label": r["label"],
            "predicted": int(pred),
            "licenses": r["licenses"],
            "correct": int(pred) == r["label"],
        })

    pos = [r for r in results if r["label"] == 1]
    neg = [r for r in results if r["label"] == 0]
    recall = sum(r["predicted"] for r in pos) / len(pos) if pos else float("nan")
    fpr = sum(r["predicted"] for r in neg) / len(neg) if neg else float("nan")
    fn = [r for r in pos if not r["predicted"]]

    print(f"\n=== Nomos real-corpus eval ===")
    print(f"  Positives : {len(pos):4d}   Recall : {recall:.4f}")
    print(f"  Negatives : {len(neg):4d}   FPR    : {fpr:.4f}")
    print(f"  False negatives: {len(fn)}")
    if fn:
        print("  FN samples (first 10):")
        for r in fn[:10]:
            print(f"    [{', '.join(r['licenses'][:2])}]  {r['path']}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {RESULTS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-only", action="store_true")
    ap.add_argument("--head", type=int, default=None, help="evaluate first N records only")
    args = ap.parse_args()

    testdata_path = clone_testdata()
    if args.clone_only:
        return

    labeled = parse_manifest(testdata_path)
    print(f"Manifest: {len(labeled)} labeled files")
    run_eval(testdata_path, labeled, args.head)


if __name__ == "__main__":
    main()
