"""
Generate small sample corpora that exercise every stage of the RefineCode
pipeline, written as JSONL under sample_data/.

  * sample_data/raw_code.jsonl  — code files: exact dups, near-dups, PII,
    copyright headers, and several low-quality files each tripping a specific
    heuristic rule, plus enough Java/HTML to show downsampling.
  * sample_data/web_docs.jsonl  — unlabelled web pages for code-data recall.
  * sample_data/web_seed.jsonl  — labelled web pages (adds int 'label') to train
    the fastText recaller.

Run:  python scripts/make_sample_data.py
"""

from __future__ import annotations

import json
import os
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "sample_data")
NOW = time.time()

GOOD_PY = '''\
import math


def area_of_circle(radius):
    """Return the area of a circle with the given radius."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * radius * radius


def circumference(radius):
    return 2 * math.pi * radius
'''

# Near-duplicate of GOOD_PY (one comment changed) -> should collapse in MinHash.
NEAR_DUP_PY = GOOD_PY.replace("Return the area", "Compute the area")

COPYRIGHT_PY = '''\
# Copyright Intel Corporation (C) 2014-2016
# Licensed under the Apache License, Version 2.0
# All rights reserved.

def running_mean(values, window):
    """Compute a simple moving average over `values` with the given window."""
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out.append(acc / window)
    return out
'''

PII_PY = '''\
# contact the maintainer for support
API_KEY = "sk-live-9f8a7b6c5d4e3f2a1b0c"
password = "hunter2-super-secret"


def connect(host="10.0.0.42"):
    admin_email = "admin@example.com"
    return (host, admin_email)


def ping():
    return connect()
'''

# Low-quality: >40% of lines are assert statements (Table 11 assert rule).
ASSERT_HEAVY_PY = '''\
def test_things():
    assert 1 == 1
    assert 2 == 2
    assert 3 == 3
    assert 4 == 4
    assert 5 == 5
    assert 6 == 6
'''

# Low-quality: placeholder spam (Table 11 placeholder rule, >1%).
PLACEHOLDER_PY = '''\
def process(data):
    # TODO: implement this
    # FIXME: handle edge cases
    # your code here
    pass


def transform(x):
    # TODO
    return x
'''

# Low-quality: does not parse into a Python AST (Table 12 AST rule).
UNPARSEABLE_PY = '''\
def broken(:
    return ???
this is not python at all !!!
'''

# Low-quality: import-heavy, >30% import lines (Table 12 import rule).
IMPORT_HEAVY_PY = '''\
import os
import sys
import json
import math
import time
import re
x = 1
'''

# Low-quality code file: a single very long minified line (general-code guard).
MINIFIED_JS = "var a=1;" + "".join(f"f{i}();" for i in range(400))

# Hex-heavy file (Table 11 hex rule).
HEX_BLOB = "const blob = [" + ",".join("0xDEADBEEF" for _ in range(200)) + "];"

GOOD_JAVA = '''\
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
'''

GOOD_HTML = '''\
<!DOCTYPE html>
<html>
  <head><title>Docs</title></head>
  <body>
    <h1>Getting started</h1>
    <p>Install the package and import it in your project.</p>
  </body>
</html>
'''


def rec(content, path, stars, source="github", **extra):
    d = {"content": content, "path": path, "stars": stars,
         "commit_time": NOW - stars * 3600, "source": source}
    d.update(extra)
    return d


def build_raw():
    rows = [
        rec(GOOD_PY, "repo_a/geometry.py", stars=500),
        rec(GOOD_PY, "repo_b/geometry_copy.py", stars=10),        # exact dup (lower stars)
        rec(NEAR_DUP_PY, "repo_c/geo_variant.py", stars=3),        # near dup
        rec(COPYRIGHT_PY, "repo_d/mathutils.py", stars=42),
        rec(PII_PY, "repo_e/client.py", stars=15),
        rec(ASSERT_HEAVY_PY, "repo_f/test_all.py", stars=2),       # dropped: assert rule
        rec(PLACEHOLDER_PY, "repo_g/stub.py", stars=1),            # dropped: placeholder rule
        rec(UNPARSEABLE_PY, "repo_h/broken.py", stars=0),          # dropped: AST rule
        rec(IMPORT_HEAVY_PY, "repo_i/imports.py", stars=4),        # dropped: import rule
        rec(MINIFIED_JS, "repo_j/bundle.min.js", stars=7),         # dropped: long line
        rec(HEX_BLOB, "repo_k/blob.c", stars=3),                   # dropped: hex rule
    ]
    # Enough *distinct* Java / HTML files to visibly show downsampling
    # (keep ~0.49 Java / ~0.30 HTML) without them collapsing in fuzzy dedup.
    for i in range(20):
        body = "\n".join(f"        step{j}({i});" for j in range(i % 5 + 1))
        java = (f"public class Task{i} {{\n"
                f"    public void run() {{\n{body}\n    }}\n"
                f"    public int id() {{ return {i}; }}\n}}\n")
        rows.append(rec(java, f"jrepo/Task{i}.java", stars=i))
    for i in range(20):
        items = "\n".join(f"      <li>Section {i}.{j}: details about topic {j}.</li>"
                          for j in range(i % 4 + 2))
        html = (f"<!DOCTYPE html>\n<html>\n  <head><title>Guide {i}</title></head>\n"
                f"  <body>\n    <h1>Chapter {i}</h1>\n    <ul>\n{items}\n    </ul>\n"
                f"  </body>\n</html>\n")
        rows.append(rec(html, f"hrepo/page{i}.html", stars=i))
    return rows


def build_web():
    code_pages = [
        ("How do I reverse a list in Python? Use lst[::-1] or reversed(lst). "
         "def reverse(lst): return lst[::-1]. This returns a new list.", 1),
        ("Traceback (most recent call last): File main.py line 3, TypeError: "
         "unsupported operand. Fix by casting int(x) before adding.", 1),
        ("Install via pip install requests, then import requests and call "
         "requests.get(url). Handle the response status_code in your function.", 1),
    ]
    noncode_pages = [
        ("The weather today is sunny with a gentle breeze. Perfect for a walk in "
         "the park with friends and family this afternoon.", 0),
        ("Our restaurant serves fresh pasta and wood-fired pizza. Book a table "
         "for dinner and enjoy the cozy atmosphere downtown.", 0),
        ("The history of the Roman empire spans many centuries of conquest, "
         "culture, and political change across the Mediterranean world.", 0),
    ]
    seed = code_pages + noncode_pages
    # Unlabelled docs to recall (mix of code-like and not, plus a filename hit).
    docs = [
        rec("def quicksort(a): return a if len(a)<2 else quicksort([x for x in a[1:] if x<a[0]])+[a[0]]",
            "web/post1.html", stars=0, source="web"),
        rec("Best hiking trails near the coast for a relaxing weekend getaway.",
            "web/post2.html", stars=0, source="web"),
        rec("numpy==1.26\nrequests>=2.31\npandas", "project/requirements.txt", stars=0, source="web"),
    ]
    return seed, docs


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "raw_code.jsonl"), "w", encoding="utf-8") as fh:
        for r in build_raw():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    seed, docs = build_web()
    with open(os.path.join(OUT, "web_seed.jsonl"), "w", encoding="utf-8") as fh:
        for content, label in seed:
            fh.write(json.dumps({"content": content, "path": "seed", "label": label}) + "\n")
    with open(os.path.join(OUT, "web_docs.jsonl"), "w", encoding="utf-8") as fh:
        for r in docs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote sample data to {OUT}/")


if __name__ == "__main__":
    main()
