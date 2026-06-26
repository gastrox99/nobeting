---
name: Nöbet scheduling gotchas
description: Non-obvious correctness traps in the shift-scheduling app (display caching, weekend tracking)
---

# A schedule display cache must invalidate on identity, not size
The grid/list display cache must be keyed by the full identity of the inputs
(year, month, team, columns), never by comparing only list lengths or column names.

**Why:** Two different months can share the same day count AND the same starting
weekday, producing identical column labels and lengths. A size-only check then
silently shows the previous month's schedule — a hard-to-spot data-correctness bug.

**How to apply:** Whenever you add a cache tied to the schedule, store a tuple key of
all inputs the cached value depends on and compare against it; extend the key when
the cache gains new inputs.

# Weekend "consecutiveness" needs a monotonic week index, not ISO week number
The penalty for back-to-back weekend shifts compares a per-day week value with
`+ 1`. That value must be a strictly monotonic week index, computed from the
Monday-of-week ordinal (`(date.toordinal() - weekday) // 7`).

**Why:** Two earlier attempts failed: (1) `(day-1)//7` blocks split a real Sat/Sun
weekend across buckets; (2) raw ISO week number resets at the year boundary
(…52, 53, 1), so January's first vs. second weekend differ by a huge negative jump
and consecutive weekends go undetected. The Monday-ordinal index keeps Sat+Sun in
the same bucket AND increments by exactly 1 every calendar week across year ends.

**How to apply:** Every builder of the day-details structure must populate this same
monotonic `week` field, or the algorithm silently falls back to broken math.
Regression coverage: a January-year-boundary case (e.g. 2022) is the key test.
