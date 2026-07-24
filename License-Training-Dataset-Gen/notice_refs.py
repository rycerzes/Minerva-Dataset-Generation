"""Build a NOTICE / short-form reference layer from ScanCode license rules.

The Atarashi reference index currently holds only full license *bodies*, but real
queries are short *notices* ("Licensed under the Apache License, Version 2.0 ...").
ScanCode ships ~30k short rule texts (notice / reference / tag / intro) each tagged
with a license_expression — exactly the missing register. We extract the
single-license ones (skip compound AND/OR/WITH), strip the {{...}} template markers,
and cache them keyed to the license.

  from notice_refs import load_notice_refs
  units = load_notice_refs()          # list[(license_key, text)]
"""
import json
import re
from pathlib import Path

CACHE = Path("cache/notice_refs.json")
_MARK = re.compile(r"\{\{.*?\}\}", re.DOTALL)   # strip ScanCode template markers
_COMPOUND = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)
# rule types that represent the short-form register we want as references
_KINDS = ("is_license_notice", "is_license_reference", "is_license_tag", "is_license_intro")


def build_notice_refs(min_chars=15, max_chars=3000):
    from licensedcode.models import load_rules
    import licensedcode
    base = Path(licensedcode.__file__).parent / "data" / "rules"
    out = []
    for r in load_rules(base):
        expr = (r.license_expression or "").strip().lower()
        if not expr or _COMPOUND.search(expr):     # single-license units only
            continue
        if not any(getattr(r, k, False) for k in _KINDS):
            continue
        txt = _MARK.sub(" ", r.text() if callable(getattr(r, "text", None)) else getattr(r, "text", ""))
        txt = " ".join(txt.split())
        if min_chars <= len(txt) <= max_chars:
            out.append((expr, txt))
    return out


def load_notice_refs():
    if CACHE.exists():
        return [tuple(x) for x in json.loads(CACHE.read_text())]
    units = build_notice_refs()
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(units))
    return units


if __name__ == "__main__":
    from collections import Counter
    units = load_notice_refs()
    keys = Counter(k for k, _ in units)
    print(f"notice reference units: {len(units)}  distinct licenses: {len(keys)}")
    print("top 10 by unit count:", keys.most_common(10))
    print("apache-2.0 units:", keys.get("apache-2.0", 0))
    for k, t in units:
        if k == "apache-2.0":
            print("  sample apache-2.0 notice:", repr(t[:120]))
            break
