"""Run the pairwise tournament: select pairs, judge them, update TrueSkill.

Position bias is neutralised by randomly assigning each speech to slot A or B per
comparison. Outcomes are checkpointed to a JSONL log so a run can resume.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..config import cfg
from ..macro.uk_macro import MacroContext
from ..schema import Speech
from .engine import Tournament


def run_tournament(
    speeches: list[Speech],
    judge,
    *,
    appearances_per_speech: int | None = None,
    macro: MacroContext | None = None,
    log_path: str | Path = "data/processed/tournament_log.jsonl",
    seed: int = 0,
    concurrency: int | None = None,
    resume: bool = False,
    full_run: bool = False,
    anonymizer=None,
    min_total: int | None = None,
    max_new: int | None = None,
) -> Tournament:
    """min_total: keep going until the log holds at least this many comparisons (a
    full re-run resumed in chunks). max_new: stop after this many new comparisons in
    this call, so a chunk fits a time limit; the log makes the next call resume."""
    tcfg = cfg()["tournament"]
    jcfg = cfg()["judge"]
    appearances = appearances_per_speech or tcfg["target_appearances_per_speech"]
    concurrency = concurrency or jcfg.get("concurrency", 8)
    macro = macro or MacroContext()

    by_id = {s.id: s for s in speeches}
    ids = list(by_id)
    tour = Tournament(
        ids,
        initial_mu=tcfg["initial_mu"],
        initial_sigma=tcfg["initial_sigma"],
        seed=seed,
        draw_probability=tcfg.get("draw_probability", 0.0),
    )
    # A Jev verdict carries a calibrated confidence. Below the threshold the model
    # could not really separate the two documents, and recording that as a full win
    # feeds TrueSkill a near coin-flip as if it were decisive; it is recorded as a
    # draw instead (split-half reliability 0.54 -> 0.67, scripts/draw_experiment.py).
    # Chat-judge records carry no calibrated probability ("p_a"), so they never draw.
    draw_below = tcfg.get("draw_below_confidence")

    def drawn(rec: dict) -> bool:
        return bool(draw_below) and "p_a" in rec and rec.get("confidence", 1.0) < draw_below
    macro_str = {sid: macro.string(by_id[sid].date) for sid in ids}
    rng = random.Random(seed)
    if anonymizer is None:
        from ..process.anonymize import Anonymizer
        from ..process.roster import build_roster
        anonymizer = Anonymizer(build_roster([s.speaker for s in speeches]))

    total_comparisons = appearances * len(ids) // 2
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sigma_target = tcfg["sigma_convergence"]
    batch = max(concurrency * 4, 32)
    done = 0
    failures = 0

    # Resume: replay previously logged outcomes so a crashed/killed run continues
    # instead of re-paying for completed comparisons.
    if resume and log_path.exists():
        replayed = 0
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    a_id, b_id, w = rec["a"], rec["b"], rec.get("winner")
                except (json.JSONDecodeError, KeyError):
                    continue
                if a_id not in by_id or b_id not in by_id:
                    continue  # belongs to a different pool
                if w == "A":
                    tour.record(a_id, b_id, drawn=drawn(rec)); replayed += 1
                elif w == "B":
                    tour.record(b_id, a_id, drawn=drawn(rec)); replayed += 1
        done = replayed
        print(f"[tournament] resumed: replayed {replayed} logged comparisons")
        logf = log_path.open("a", encoding="utf-8")
    else:
        logf = log_path.open("w", encoding="utf-8")
    consecutive_failures = 0
    pbar = tqdm(total=total_comparisons, desc=f"tournament[{judge.model}]")

    def judge_pair(pair):
        i, j = pair
        # randomize slot assignment to cancel position bias
        if rng.random() < 0.5:
            a_id, b_id = i, j
        else:
            a_id, b_id = j, i
        a, b = by_id[a_id], by_id[b_id]
        # anonymised lazily — only documents actually drawn pay for it — and never
        # with a fallback to the raw, named text
        res = judge.compare(
            anonymizer.text_of(a), macro_str[a_id],
            anonymizer.text_of(b), macro_str[b_id],
            a_type=a.source_type, b_type=b.source_type,
        )
        return a_id, b_id, res

    # Budget the comparisons to the GENUINELY new speeches (those with no logged
    # comparisons after the replay): ~appearances/2 comparisons per new speech. This
    # gives a fresh full run its full ~15*N budget (all speeches new) and keeps a daily
    # increment small and always-terminating - it never tries to re-converge speeches
    # that already carry a rating (which, if a few can't reach the target, would loop
    # forever dumping comparisons onto the rest).
    #
    # full_run instead budgets to the whole pool's target (15*N at 30 appearances),
    # counted from zero. That is what makes a killed full run resumable: the "new"
    # rule would see every speech carrying one or two comparisons and stop dead,
    # leaving the pool under-sampled.
    new_count = sum(1 for c in tour.n_comp.values() if c == 0)
    budget = done + appearances * new_count // 2
    if full_run:
        budget = max(budget, total_comparisons)
    if min_total:
        budget = max(budget, min_total)
    cap = max(done, min_total or 0) + appearances * len(ids)   # hard safety bound
    if max_new is not None:
        budget = min(budget, done + max_new)
    while done < budget and done < cap:
        n = min(batch, budget - done)     # a chunk stops at its budget, not a batch past it
        pairs = tour.select_pairs(n, tcfg["pairing"])
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(judge_pair, p) for p in pairs]
            for fut in as_completed(futs):
                try:
                    a_id, b_id, res = fut.result()
                except Exception as e:  # exhausted retries on this pair -> skip
                    failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= 200:
                        logf.close(); pbar.close()
                        raise RuntimeError(
                            f"Aborting: {consecutive_failures} consecutive judge "
                            f"failures (API likely down). Last error: {e!r}")
                    continue
                consecutive_failures = 0
                w = res.get("winner")
                if w == "A":
                    tour.record(a_id, b_id, drawn=drawn(res))
                elif w == "B":
                    tour.record(b_id, a_id, drawn=drawn(res))
                else:
                    continue  # unparseable -> skip, don't corrupt ratings
                logf.write(json.dumps({"a": a_id, "b": b_id, **res}) + "\n")
                logf.flush()
                done += 1
                pbar.update(1)
        if tour.max_sigma() < sigma_target and tour.mean_appearances() >= appearances:
            break

    logf.close()
    pbar.close()
    if failures:
        print(f"[tournament] completed with {failures} skipped comparisons "
              f"(transient API errors)")

    for sid, s in by_id.items():
        r = tour.rating(sid)
        s.mu, s.sigma, s.n_comparisons = round(r.mu, 3), round(r.sigma, 3), tour.n_comp[sid]
    return tour
