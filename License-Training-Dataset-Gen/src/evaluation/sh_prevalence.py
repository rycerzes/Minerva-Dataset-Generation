"""Prevalence-weighted tail benchmark.

The one-blob-per-licence benchmark is macro by construction: `GPL-3.0-only` and
`GPL-3.0-or-later` each get one blob although real bodies split 107:4. That makes it
the right instrument for "which licences can we identify at all" and the wrong one for
"how often are we right on real files" — and reading it as the latter is how the
only/or-later rule got judged twice, wrongly, in both directions.

This is the companion. The Software Heritage 20k sample is a *random* draw from 6.9M
real licence blobs, so its label distribution is the prevalence. Blobs used for
training are excluded (the whole cap-12 pool, a superset of what was trained on).

Labels are ScanCode's, so this measures our agreement with ScanCode on a
prevalence-representative sample. It is not a head-to-head and cannot show a win over
it; what it shows is how a change lands on real files rather than on rare variants.
"""
import collections, csv, io, json, os, pathlib, random, subprocess, sys, tempfile
import pandas as pd
from atarashi.agents.cascade import Cascade
from atarashi.spdx.resolver import lookup_shortname, shortname_index

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = str(os.environ.get("SH_DATA_DIR", ROOT / "cache" / "sh"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 0

held = {c["sha1"] for c in json.load(open(f"{S}/sh_train_corpus.json"))}
have = {f: os.path.join(r, f) for r, _, fs in os.walk(f"{S}/blobs20k") for f in fs}
proc = subprocess.Popen(["zstd", "-dc", f"{S}/blobs-scancode.csv.zst"], stdout=subprocess.PIPE)
pool = []
for row in csv.DictReader(io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")):
    if row["sha1"] in held or row["sha1"] not in have:
        continue
    if not row["license"] or " " in row["license"]:
        continue
    try:
        if float(row["score"]) < 95.0:
            continue
    except ValueError:
        continue
    pool.append((row["sha1"], row["license"]))
proc.stdout.close(); proc.wait()
random.Random(0).shuffle(pool)
pool = pool[:N]
print(f"prevalence sample: {len(pool)} blobs, {len(set(l for _, l in pool))} licences "
      f"(training blobs excluded)", flush=True)
top = collections.Counter(l for _, l in pool).most_common(5)
print(f"  most frequent: {top}")

from evaluation.agent import load_license_list
df = load_license_list()
idx = shortname_index(df["shortname"])
kw = {"top_k": TOP_K} if TOP_K else {}
agent = Cascade(df, use_gate=False, **kw)

fd, path = tempfile.mkstemp(suffix=".txt"); os.close(fd)
n = ans = hit = ex = unnameable = 0
per_lic = collections.defaultdict(lambda: [0, 0])
for sha, lic in pool:
    want = lookup_shortname(lic, idx)
    if not want:
        unnameable += 1
        continue
    n += 1
    try:
        open(path, "w").write(open(have[sha], errors="replace").read())
    except OSError:
        continue
    res = agent.scan(path)
    names = {r["shortname"] for r in res if r["shortname"] != "UNKNOWN"}
    if not names:
        continue
    ans += 1
    ok = res[0]["shortname"] == str(want)
    hit += ok
    ex += names == {str(want)}
    per_lic[lic][0] += ok; per_lic[lic][1] += 1
os.unlink(path)

macro = sum(c / t for c, t in per_lic.values()) / len(per_lic) if per_lic else 0
print(f"\nscored {n} blobs ({unnameable} labelled with a licence the agent cannot name)")
print(f"  coverage           {ans}/{n} = {ans/n:.4f}")
print(f"  R@1 (prevalence)   {hit}/{n} = {hit/n:.4f}   precision {hit/max(ans,1):.4f}")
print(f"  exact-set          {ex}/{n} = {ex/n:.4f}")
print(f"  macro over {len(per_lic)} licences {macro:.4f}  <- what the 66-blob benchmark measures")
