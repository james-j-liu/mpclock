"""Remember pages that yielded nothing, so the daily run stops re-fetching them.

The Bank's sitemap lists pages before they have content — minutes and Report pages
for meetings months away, speeches marked "text to be published Monday" — and a
few pages never yield a usable document. Without a memory, every morning fetched
all of them again. A URL that misses is retried daily for its first two weeks
(a placeholder normally fills within days), then weekly, indefinitely; the moment
it yields a record it leaves the list.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

DAILY_FOR_DAYS = 14
THEN_EVERY_DAYS = 7

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
# the Bank's URLs carry the month of the event: ".../2026/november/..." (speeches)
# or ".../november-2026" (minutes, Reports)
_MONTH_RE = re.compile(
    r"/(?P<y1>\d{4})/(?P<m1>%s)(?:/|$)|/(?P<m2>%s)-(?P<y2>\d{4})(?:/|$)"
    % ("|".join(_MONTHS), "|".join(_MONTHS)))


def url_month(url: str) -> tuple[int, int] | None:
    """(year, month) an event URL is dated to, or None if it carries no month."""
    m = _MONTH_RE.search(url.lower())
    if not m:
        return None
    return (int(m.group("y1") or m.group("y2")),
            _MONTHS.index(m.group("m1") or m.group("m2")) + 1)


class Misses:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.state: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}
        self.today = _dt.date.today()

    def due(self, url: str) -> bool:
        """Whether a (possibly previously missed) URL should be fetched today."""
        rec = self.state.get(url)
        if not rec:
            return True
        # The Bank lists a meeting's minutes/Report page in its sitemap weeks before
        # the meeting, so "first seen" ages it into weekly retries before the day it
        # actually publishes (5 Nov 2026 was skipped this way). A page dated to this
        # or last month is always worth a daily look.
        ym = url_month(url)
        if ym:
            t = self.today
            prev = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
            if ym in ((t.year, t.month), prev):
                return True
        first = _dt.date.fromisoformat(rec["first"])
        last = _dt.date.fromisoformat(rec["last"])
        if (self.today - first).days < DAILY_FOR_DAYS:
            return True
        return (self.today - last).days >= THEN_EVERY_DAYS

    def update(self, tried: list[str], found: set[str]) -> None:
        today = self.today.isoformat()
        for url in tried:
            if url in found:
                self.state.pop(url, None)
            else:
                rec = self.state.setdefault(url, {"first": today, "last": today})
                rec["last"] = today

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=0, sort_keys=True), encoding="utf-8")
