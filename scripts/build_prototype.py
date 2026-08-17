"""Inline the data files into prototype/build.html -> prototype/index.html.

build.html is the SOURCE and carries four placeholders; index.html is the built,
self-contained artifact (no external requests, so it survives the artifact CSP).

Run:  py -3.10 build_prototype.py     (any Python 3 works, no deps)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "prototype", "build.html")
OUT = os.path.join(HERE, "prototype", "index.html")
BLOCKS = [
    ("__DATA__",   "track1.json"),
    ("__STRUCT__", "track1_struct.json"),
    ("__STRUCT2__", "track1_struct2.json"),
    ("__POCKET__", "track1_pocket.json"),
    ("__CORR__",   "track1_correspondence.json"),
]

def main():
    html = open(SRC, encoding="utf-8").read()
    for token, fn in BLOCKS:
        path = os.path.join(HERE, "data", fn)
        if token not in html:
            sys.exit(f"placeholder {token} missing from build.html")
        payload = open(path, encoding="utf-8").read().strip()
        # a literal </script> inside JSON would close the host tag early
        payload = payload.replace("</", "<\\/")
        html = html.replace(token, payload)
        print(f"  {token:12s} <- {fn} ({os.path.getsize(path)/1024:.0f} KB)")
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
