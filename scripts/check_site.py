"""Fail before deploying a page that cannot run.

Checks that every inline <script> in site/*.html parses (node --check) and that every
site/*.json the pages load is valid JSON. A single unescaped apostrophe in a JavaScript
string stops the whole page script, so the site renders nothing.

    python scripts/check_site.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", re.S | re.I)


def main() -> int:
    bad = []
    for page in sorted(SITE.glob("*.html")):
        html = page.read_text(encoding="utf-8")
        for i, m in enumerate(SCRIPT.finditer(html)):
            attrs, code = m.group(1), m.group(2)
            if "json" in attrs.lower() or not code.strip():
                continue
            line = html[:m.start(2)].count("\n")
            suffix = ".mjs" if "module" in attrs.lower() else ".js"
            with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
                f.write("\n" * line + code)     # keep line numbers matching the page
            r = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
            Path(f.name).unlink()
            if r.returncode:
                err = r.stderr.replace(f.name, f"{page.name}")
                bad.append(f"{page.name} script {i + 1}:\n{err.strip()}")
    for data in sorted(SITE.glob("*.json")):
        try:
            json.loads(data.read_text(encoding="utf-8"))
        except ValueError as e:
            bad.append(f"{data.name}: invalid JSON ({e})")
    for b in bad:
        print(b, file=sys.stderr)
    print(f"site check: {'FAILED' if bad else 'ok'} ({len(list(SITE.glob('*.html')))} pages, "
          f"{len(list(SITE.glob('*.json')))} data files)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
