# DRAFT pre-registration 1 (v6): crash ladder on C4's idle cash (S1 / combination P1)

Status: DRAFT v6 for owner approval (revised 2026-11-09 after the round-5 quant-critic review
`review_draft1_v5.md`, verdict FIX-REQUIRED with one sentence; earlier rounds `review_draft1_v4.md`,
`review_draft1_v3.md`, `review_draft1_v2.md`, `review_draft1.md`).
Nothing has been run and no payoff has been computed by anyone. Once approved, commit unchanged before the first
backtest line executes.

Policy context (`decisions.md`, 2026-11-09): real money is spot-only on Tokocrypto, no margin, no
futures, no leverage. This is the one candidate that may go to paper and then real money.

## Changes v6 (round-5 item -> change)

No threshold was moved.

| R5 item | Change in v6 |
|---|---|
| 1 | Clock-convention sentence added verbatim (Fill model and timing); exit-eligibility and PERCENT_PRICE-retry wording aligned with it. |
| Note: K | Kill-rule K = 0.8 x the forward book's start equity; realised ladder PnL net of fees. |
| Note: window | Next-day fill window written as "bars 00:11 to 00:09". |
| Note: counts | Stale counts updated to the compounding figures (H-cascade ~26; G4(a) 19 / 17), fixed-$1,000 figures kept in parentheses. |
| Note: missing bars | A missing 1m bar is replaced by the next available bar. |
| Note: contamination | Round-5 line added (exit-crossing times for the 18 2025-10-10 fills were printed, not recorded). |

## Changes v5 (round-4 item -> change)

No threshold was moved.

| R4 item | Change in v5 |
|---|---|
| 1 | Added (critic's sentence): "An episode, with all its fills, belongs to the calendar quarter and the period (DEV or HOLDOUT) of its first fill." The "no episode crosses a quarter boundary" line is now qualified as holding only at a fixed $1,000. |
| 2 | Kill-rule ladder DD redefined on S_t = K + cumulative realised ladder PnL + unrealised PnL of open ladder positions at the daily close, K = ladder capital of the first forward run; stop when S_t falls more than 20% below its running peak. |
| 3 | Fill detection (critic's sentence): "A fill in the 1m bar opening at hh:mm is detected at the first poll at or after hh:mm+1." |
| Notes | Disclosure adds 58/26 (>= 10 coins: 19/17) and the straddling TRB episode, explained by $5-rule compounding; G4(c) states the outage-covers-time-stop clause cannot fire; run-end wording (10-31 bids rest to 11-01 00:10, exits on data through 11-06, "last close" = 11-06 close); G8 uses the H-ladder return; Step 2 wording fixed (C4 can enter a coin the ladder holds); two rare unmodelled cases stated; G5(b) says "re-run". |

## Changes v4 (round-3 item -> change)

| R3 item | Finding | Change in v4 |
|---|---|---|
| 1 | C4 sizing base inside the combined book unstated (live fund sizes from cash + C4 holdings only) | C4 sizes from TOTAL book equity (cash + C4 holdings + ladder holdings); its buys are trimmed to available cash, so its targets equal standalone C4's and the ladder is a pure add-on. Separate per-coin quantities per sleeve. C4 fills at the 00:00 daily open in both books. Reference replay: the `fund/report.py` `compare` loop. Note added: a live implementation must change `plan_orders`' equity input to match; not done now. |
| 2 | Continuous run vs fresh start at HOLDOUT | One continuous run 2018-08-17 to 2026-10-31 from USD 1,000; DEV and HOLDOUT are slices; max DD computed within each slice. |
| 3 | G4 mechanics underspecified | (a) the >= 10-coin episode set comes from the primary run; (b) fills and denominators held fixed, only exits repriced; (c) time stop runs from the fill (fill + 72h); if the outage covers that moment, exit at the first poll after the outage ends, at that bar's open minus 2%; (d) 2% replaces the 0.5%. Affected: 15 of 49 DEV H-ladder and 14 of 23 H-cascade episodes (MEASURED by the critic). |
| 4 | Data not pinned; repo cache ends 2026-10-10 with no high/low | Fetch daily OHLCV and 1m data through 2026-11-06; freeze a file manifest (path, size, SHA-256) committed before the first run; the repo daily cache is not used. Series-end rule (c) = "the last bar, if it is before the data end date". |
| 5 | Episode denominator ambiguous | Denominator = the ladder capital of the daily run that placed the bid behind the episode's first fill. |
| 6 | G0 can pass by coincidence | Only a NEW price level inside a Binance spread wider than one tick (use a less liquid pair if needed), and the level must show the order's exact quantity. Fallback removed. |
| 7 | Universe lag wording | "Shift the volume and eligibility inputs by one day; membership takes effect on the 1st." The membership mask itself is not shifted. |
| Note | Counts "can only lower" | Reworded: the minute-clock rules may move counts in either direction. Round-3 counts added to the disclosure. |
| Note | Kill-rule DD series unnamed | Ladder DD measured on ladder-sleeve equity = allocated ladder capital + realised ladder PnL + open ladder positions marked at the daily close. |

## Changes v3 (round-2 item -> change)

| R2 item | Finding | Change in v3 |
|---|---|---|
| 1 | Episode merge rule undefined; it alone moves H-cascade across G1 | Episode = connected component of fills: a fill joins an episode if it is within 72h of that episode's LAST fill. H-cascade episode = an H-ladder episode with at least one -30% or -45% fill. Expected DEV counts stated (~49 / ~23). G1 minimum for H-cascade kept at 20; no threshold moved after the counts were seen (disclosed). |
| 2 | Series end misses plain delistings; universe helpers count bars across gaps | Series also ends at the instrument's last bar (MATIC, FTM, RNDR, ERD, TON named). Stated wrapper: every series is split into segments before `top_by_volume` / eligibility, each segment a separate column. One-day universe lag stated. |
| 3 | G4 (haircut 50% vs 100%) is empty | G4 replaced by a tail test: G2 must still pass, for every hypothesis, with 2% time-stop slippage AND a 48h exit outage on every episode with >= 10 coins filled. 100% haircut kept as information only. |
| 4 | Halt semantics and combined-book C4 path undefined | Primary: both books run with NO drawdown halt; each book's max DD and the dates the live sticky 30% halt would have fired are reported. Sensitivity: both books under the live sticky halt (never cleared, blocks C4 buys too). C4 weights inside the combined book come from the combined book's own path. |
| 5 | Timing details | Buy fills count from the 00:11 bar; exits eligible from the bar after the poll minute; exit target anchored to the open that set the bid; scan also flags next-day lows <= that open x 0.80; 1m window covers >= 6 days after each fill. |
| 6 | Outage/missed-run sensitivities optimistic; paper run executed at ~02:27 UTC | Outage now targeted at broad episodes inside the G4 tail test. Late-run sensitivity (bids placed 02:30 UTC). New real-money precondition: poller uptime demonstrated in forward paper (criterion stated). |
| 7 | H-cascade does not isolate cascades (9 of 26 single-coin) | Breadth split pre-specified: market-wide = BTC daily low <= open x 0.90 on any episode day, OR >= 5 T20 coins at -20% within the episode. The mechanism claim requires G2 on the market-wide H-cascade subset. Single-coin episodes reported separately. |
| 8 | G0 has no procedure | G0 procedure and pass criterion written (order must appear in Binance public depth and fill on Binance prints). PERCENT_PRICE_BY_SIDE modelled: exit sells above 2x the 5-min average price are rejected and retried at later polls. |
| 9 | Bootstrap details | G2: fixed calendar quarters as blocks, pooled-mean statistic, empty-draw handling, seed. G6: paired stationary bootstrap specified as new code with a unit test. |
| 10 | "Without 2025-10-10" ambiguous | Defined: no ladder bids for the whole episode containing any 2025-10-10 fill. Applies to G5 and G7. |
| 11 | Disclosure | Round-2 counts (fixes 1, 3, 7 and the others) added as seen. |

<details><summary>Changes v2 (round-1 item -> change), kept for the audit trail</summary>

| R1 item | Change in v2 |
|---|---|
| 1 | Series ends at gaps > 24h / symbol re-use; position marked at last pre-cut close minus 50%; LUNA after 2022-05-13 and FTT gap excluded; migrated symbols restart history (extended in v3 item 2). |
| 2 | Always-on 15-min poller as primary exit; no exit in the entry bar; sells need a print >= 0.1% above target; once-daily exit and 48h outage as sensitivities. |
| 3 | Fills only after the 00:10 run; tiers below market at 00:10 skipped; missed runs leave bids resting. |
| 4 | Ladder capital = max(0, 0.8 x (equity - C4 held - open ladder)); fixed 1/20 slots; no bid on held coins; C4 held weights from the replay; bids lock cash. |
| 5 | $1,000 start capital, $5 minimum applied and counted, $10,000 as information. |
| 6 | 72h-merged episodes; episode return = PnL / ladder capital; H-ladder vs H-cascade; per-tier minimums. |
| 7 | Single-best-episode drop gate; per tier and per coin. |
| 8 | Holdout gates with and without 2025-10-10. |
| 9 | Materiality hurdle renamed; paired Sharpe-difference bootstrap; daily-close marks; unmodified C4 baseline. |
| 10 | 0.1% trade-through for buys; G0 routing precondition. |
| 11 | Per coin, short-membership PnL share, Tokocrypto-listed share. |
| 12 | Contamination disclosure widened. |
| 13 | Fund guards, separate forward ledger, maker-fee assumption, 2% time-stop sensitivity. |

</details>

## Hypotheses

- **H-cascade (the mechanism claim).** In market-wide liquidation cascades, forced sellers push
  liquid coins far below fair value for minutes to hours; resting spot bids at -30% and -45% below
  the day's open earn the rebound. Unit: H-cascade episodes; return = PnL of the -30% and -45%
  fills in the episode / the episode denominator (defined under Episodes).
- **H-ladder (the deployable package).** The full ladder (-20/-30/-45%) has positive net return per
  episode, and adding it to C4 x2 + PAXG on C4's idle cash improves the combined book.

Adoption requires H-ladder AND H-cascade AND H-cascade on the market-wide subset (G2c). If only
H-ladder passes, the result says "buying ordinary -20% alt days pays", a different mechanism (close
to mean reversion, already weak here: RSI2 DEV Sharpe 0.09); reported, not adopted. If H-cascade
passes overall but not on the market-wide subset, the "liquidation cascade" story is not supported
(the profit would come from single-coin events such as TRB/FTT-type unwinds); reported, not adopted.

Who pays (H-cascade): liquidated leveraged longs and margin-called holders, via exchange
liquidation engines, while market makers withdraw.

## Episodes (round-2 item 1, exact)

- A **fill** is one filled tier on one coin.
- Fills are sorted by time. An episode starts with a fill; each next fill joins the current episode
  if it occurs within 72 hours of the episode's LAST fill so far; otherwise it starts a new episode.
  (Connected components with chaining.)
- **H-ladder episode:** every episode. **H-cascade episode:** an episode containing at least one
  -30% or -45% fill.
- **Market-wide** episode: on any UTC day spanned by the episode (first fill day to last fill day),
  BTCUSDT's daily low <= its daily open x 0.90, OR at least 5 distinct T20 coins had a daily low
  <= daily open x 0.80 on days spanned by the episode (counted from daily bars, whether or not the
  ladder filled them). Otherwise **single-coin/narrow**.
- Episode return = net PnL of the episode's fills (for H-cascade, of its -30%/-45% fills only)
  / **episode denominator** = the ladder capital computed by the daily run that placed the bid
  behind the episode's first fill (round-3 item 5).
- **Expected counts (MEASURED by the round-2 and round-3 critics from daily lows, i.e. a daily
  approximation of the rules):** DEV 49 H-ladder / 23 H-cascade; HOLDOUT 10 / 3. Market-wide DEV
  H-cascade episodes: 17, spread over 13 quarters (13 of the 17 meet the BTC condition). The
  minute-clock rules (00:11 start, 0.1% trade-through, 72h chaining on minute timestamps, fills in
  the next day's 00:00-00:09 bars on the previous day's bids) may move these counts in EITHER direction:
  minute-level chaining can split episodes, and early-morning fills can add some.
- The counts above assume a fixed USD 1,000 for the $5 rule. Under compounding (the actual run) the
  $5 minimum binds less often: round-4 critic, with the $5 rule scaled to the approximate C4 x2
  equity path, measured DEV 58 H-ladder / 26 H-cascade (>= 10 coins: 19 / 17). G1 therefore has more
  margin; the thresholds were NOT moved.
- **An episode, with all its fills, belongs to the calendar quarter and the period (DEV or HOLDOUT)
  of its first fill.** (Round-4 item 1; under compounding the TRB episode of 2023-12-31..2024-01-03
  straddles 2023Q4/2024Q1.)

## Event counts already seen (disclosure)

Round 1 (critic, daily low < open x tier, T20 only, no payoff computed):

| | tier -20% | tier -30% | tier -45% |
|---|---|---|---|
| DEV coin-days / UTC-day clusters / 72h episodes | 535 / 99 / 65 | 158 / 44 / 29 | 42 / 12 / 7 |
| HOLDOUT (to 2026-10-31) | 53 / 14 / 11 | 21 / 3 / 3 | 16 / 2 / 2 |

- DEV -45% tier: 30 of 42 coin-days on 2020-03-12 and 2021-05-19; 6 of 12 days are LUNA.
- HOLDOUT: 39 of 53 coin-days on 3 days (incl. 2025-02-03 and 2025-10-10); the -45% tier is
  2025-10-10 plus one ZEC day. T20 averages 2.2 members in 2018 and 11.6 in 2019 (DEV effectively
  starts in 2020).

Round 2 (critic, MEASURED, no payoff computed):
- With v2 exclusions (LUNA after 2022-05-13, coins C4 holds, $5 minimum at $1,000): DEV ~49
  H-ladder / ~23 H-cascade episodes.
- H-cascade count is 29 / 25 / 23 when merged days may be at most 1 / 2 / 3 days apart.
- 9 of 26 DEV H-cascade episodes (before the $5 rule) are a single coin with BTC near flat (TRB, FTT,
  PEOPLE, WLD, BAND, HOT, XRP, FIL).
- Among the 100 coins ever in T20, the only gaps > 24h are LUNA (a member; last close 0.00005) and
  FTT (not a member); no trigger falls within 7 days before any delisting.
- T20 members whose series ended by delisting/migration: MATIC (2024-09-10), FTM (2025-01-13), RNDR
  (2024-07-22), ERD (2020-08-31), TON (2026-06-30).
- DEV spans 26 calendar quarters (2018Q3-2024Q4); 14 of them contain H-cascade episodes.
- The fund's paper ledger has 1 run since 2026-11-06, executed at ~02:27 UTC.
- Tokocrypto: 19 of 20 current T20 coins listed as USDT pairs (ZEC is not); 78 of the 100 coins
  ever in T20 are listed today. Tokocrypto has the same PERCENT_PRICE_BY_SIDE filter as Binance.
- Lagging the universe by one day changed no trigger.

Round 3 (critic, MEASURED from daily lows, no payoff computed):
- DEV 49 H-ladder / 23 H-cascade, HOLDOUT 10 / 3, with the stated rules.
- LUNAUSDT#2 and FTTUSDT#2 never enter T20.
- Market-wide DEV H-cascade episodes: 17 over 13 quarters; 13 meet the BTC condition; counting
  breadth per day or across the episode flips none.
- At a fixed USD 1,000, no episode crosses a quarter boundary or period end; the longest spans 7
  days. (This holds only at fixed $1,000; see round 4.)
- The 2025-10-10 HOLDOUT episode is that one day only; without it, G5(b) rests on 9 episodes,
  mostly single-coin.
- The G4 delay hits 15 of 49 DEV H-ladder and 14 of 23 DEV H-cascade episodes.
- Shifting the membership mask (instead of its inputs) would give HOLDOUT 11 rather than 10 H-ladder
  episodes.
- The repo's daily cache ends 2026-10-10 for 484 symbols and has no high/low columns.

Round 4 (critic, MEASURED from daily lows, no payoff computed):
- At a fixed USD 1,000: DEV 49 / 23 (>= 10 coins: 15 / 14); HOLDOUT 10 on this draft's lag rule.
- With the $5 rule scaled to the approximate C4 x2 equity path (~USD 5.4k at end-2023, ~USD 7.6k at
  end-2024), i.e. **because of $5-rule compounding**: DEV 58 H-ladder / 26 H-cascade (>= 10 coins:
  19 / 17), and one episode (TRB, 2023-12-31..2024-01-03, H-cascade, 8 coins) straddles a quarter
  boundary. HOLDOUT unchanged at 11 / 3 on the mask-shift reading.
- Run end: no T20 same-day trigger from 2026-09-01 to 2026-10-31; the only next-day flag is ENA on
  2026-10-29 (fill possible only in 00:00-00:10 on 10-30).
- `research/metrics.py` annualises with sqrt(365), consistent with G6.

Round 5 (critic): a probe script printed exit-crossing times for the 18 tier fills of 2025-10-10
(HOLDOUT, already contaminated); they were not recorded. MEASURED by the critic: PERCENT_PRICE binds
that day (WLD's -30% fill in the 21:16 bar had its exit rejected at the 21:30 poll and first accepted
at 21:45); no new -20% fill or episode on 2025-10-11..16, so the G5(b) suppression is unambiguous;
C4 x2 + PAXG target gross exposure is never >= 0.90 (implied ladder share 0.8 x (1 - gross): minimum
0.092, 5th percentile 0.29, median 0.60).

**These counts were seen before the G1 thresholds were confirmed.** The thresholds (>= 40
H-ladder, >= 20 H-cascade) are the v2 values, kept unchanged in v3; they were not moved in
response to the counts. Expected H-cascade is ~26 under compounding (~23 at a fixed $1,000), close
to its threshold of 20; an "insufficient evidence" outcome is a real possibility and is accepted in
advance.

## Rule (one variant, fixed)

### Universe (ex-ante, survivorship-free)
- On the 1st of each month, the top 20 Binance USDT spot pairs by trailing 90-day quote volume.
  Lag (round-3 item 7): the volume and eligibility INPUTS are shifted by one day (the 1st uses
  trailing volume and bar counts up to the previous day's close) and membership takes effect on the
  1st; the membership mask itself is not shifted. Drawn from the
  ever-listed pool with the exclusions in `research/PREREGISTRATION_UNIVERSE.md`, each with >= 365
  daily bars of the CURRENT segment.
- **Segment wrapper (round-2 item 2):** before `research/universe_study.top_by_volume` and the
  eligibility count run, every symbol's daily series is split into segments at each series end
  (definition below). Each segment is its own column (e.g. `LUNAUSDT#1`, `LUNAUSDT#2`), so trailing
  volume and the 365-bar count never span a gap or a re-use. A migrated token's new symbol
  (POL, S, RENDER, ...) starts a new column with zero history.

### Series end, halts, delistings, splices (round-1 item 1, round-2 item 2)
- A symbol's series ENDS at: (a) the first gap > 24h between consecutive bars, (b) a symbol re-use,
  or (c) **the instrument's last bar, if it is before the data end date (2026-11-06)** (plain
  delisting or migration: e.g. MATIC 2024-09-10, FTM 2025-01-13, RNDR 2024-07-22, ERD 2020-08-31,
  TON 2026-06-30). A series still trading at the data end does NOT end there. No bar after an end
  is used.
- **Run end:** the last bids are placed on 2026-10-31 and rest until the 2026-11-01 00:10 cancel;
  exits play out on data through 2026-11-06; any position still open after that is marked at the
  2026-11-06 close ("last close"), without a haircut.
- A position open at a series end is closed at the last pre-end close x (1 - 0.50). Justification:
  a gap or delisting of a top-20 coin has been terminal or near-terminal, the last close usually
  already contains the crash, and holders can typically withdraw and sell elsewhere at a discount.
  The critic measured that no trigger falls within 7 days before any delisting in T20 history, so
  this rule is expected to touch few or no positions; a 100% haircut is reported as information.
- LUNAUSDT excluded after 2022-05-13; FTTUSDT excluded from the start of its 311-day gap.

### Daily run at 00:10 UTC (order of operations)
1. Cancel all unfilled ladder bids (releases locked cash).
2. Run C4 x2 + PAXG as `fund.core.decide` would (Monday rebalance, 0.2 band), with the reference
   replay being the `fund/report.py` `compare` loop. Sizing (round-3 item 1):
   - C4 sizes from **TOTAL book equity** E = cash + C4 holdings + ladder holdings (all marked at the
     day's 00:00 open). C4's held weights are C4 holdings / E, from the combined book's own path.
     Its target weights are therefore identical to standalone C4's; the ladder is a pure add-on.
   - C4 buys are trimmed to the cash available after step 1 (cash never negative). Trimmed buys
     persist through the 0.2 band, which is why the combined path is replayed on its own.
   - Each sleeve keeps its own per-coin quantities. The ladder never bids on a coin C4 holds (step
     4), but C4 can enter a coin the ladder already holds; both can then hold it at once, and
     behaviour stays defined because the quantities are kept separate.
   - C4 trades fill at the 00:00 daily open in BOTH books (the research/report convention); only the
     ladder's cancel and placement are timed at 00:10.
   - Live note: the fund today calls `plan_orders` with equity = cash + C4 holdings
     (`fund/core.py`, `fund/run_daily.py`), which would shrink C4 while ladder positions are open.
     A live implementation must change that equity input to total book equity to match this test.
     Not done now (research only).
3. Ladder capital = max(0, 0.8 x (cash after step 2)), i.e. 0.8 x free cash; open ladder positions
   and C4 holdings are already excluded because they are not cash.
4. For each T20 coin with NO open position (ladder or C4), place GTC limit buys at O x 0.80,
   O x 0.70, O x 0.55, where O = that day's 00:00 UTC daily open. Coin slot = ladder capital / 20,
   tier weights 0.25 / 0.35 / 0.40; slots of skipped coins stay in cash.
5. A tier already below the market at placement (the 00:10 1m open is below the tier price) is
   skipped that day.
6. Orders below USD 5 are skipped and counted.
   Not modelled (both rare, stated for completeness): the $5 minimum on EXIT sells (a ladder
   position worth < $5 at exit time is assumed sellable), and PERCENT_PRICE_BY_SIDE rejecting the
   -45% bid when the 00:10 five-minute average is above 1.1 x O (then 0.55 x O < 0.5 x average).
7. No guards in the primary run (see "Drawdown halt"). Kill switch: not simulated in the backtest
   (it is a manual live control); obeyed in forward paper.

### Fill model and timing (round-2 item 5)
- **Clock convention (round 5, verbatim):** A fill's timestamp is the open time hh:mm of its 1m bar; every 72h/48h clock (time stop, episode chaining, G4 window) runs from it. Any order sent or retried at a poll at hh:mm, including a PERCENT_PRICE retry, is live from the hh:mm+1 bar. The 5-min average at a poll at hh:mm is the mean of the closes of bars hh:mm-5 to hh:mm-1.
- Buy fills count on **bars 00:11 to 00:09**: from the 00:11 1m bar of day d through the 00:09 bar of
  day d+1, before the next run's cancel at 00:10 (so fills in day d+1's 00:00-00:09 bars belong to
  day d's bids).
- Missing 1m bars: if a needed 1m bar is absent from the data, the next available bar is used (for
  fills, exits, polls, time stops and averages). A buy fills only if a 1m low <= limit
  x 0.999; fill price = limit. Resting bids lock their USDT.
- Missed run: bids keep resting at their original levels (anchored to their own O) until a run
  cancels them. Primary assumes no missed runs.

### Exit (primary: always-on poller)
- Polls at :00/:15/:30/:45. A fill in the 1m bar opening at hh:mm is detected at the first poll at
  or after hh:mm+1. The exit sell is sent at that poll and, per the clock convention, is live from
  the bar after the poll minute (poll at 00:15 -> live from the 00:16 bar). An exit can never fill in
  the entry bar.
- Target = midpoint between the fill price and **the O that set the bid** (not the current day's
  open). Fills only if a later 1m high >= target x 1.001; fill price = target.
- **PERCENT_PRICE_BY_SIDE (MEASURED on Tokocrypto, same as Binance):** a sell is rejected if its
  price exceeds 2x the 5-minute average price. If rejected, the poller retries at each later poll;
  the retry that is accepted is the first poll at hh:mm where target <= 2 x the 5-min average (mean
  of the closes of bars hh:mm-5 to hh:mm-1), and the order is live from the hh:mm+1 bar. Bids down to 0.5x the average are allowed, so the -45%
  tier is placeable.
- Time stop: if not exited 72h after the fill, sell at the open of the 1m bar after the first poll
  at or after 72h, minus 0.5% extra slippage.

### Costs
- 0.3822% fee + 0.05% slippage per side. **Assumption:** maker fee assumed equal to the measured
  taker fee. Time-stop exits +0.5% extra slippage (primary).

### Drawdown halt (round-2 item 4)
- **Primary:** BOTH books (unmodified C4 x2 + PAXG baseline, and combined) run with NO drawdown
  halt, so the comparison isolates the ladder's contribution. For each book, report max DD and the
  dates on which the live rule (30% DD from peak, sticky: never cleared in code, `fund/run_daily.py`,
  and it blocks C4 buys as well as ladder bids) would have fired.
- **Sensitivity:** both books under the live sticky halt (fires at 30% DD, never clears, blocks all
  new buys in that book). Reported as information; it cannot change a gate.
- The 50%-of-equity order cap is never binding (<= 1.6% per tier) and is stated for completeness.

### Capital
- Start equity USD 1,000 (fund paper capital); USD 10,000 reported as information. $5 skips reported
  by tier and as a share of attempted orders.

## Data

- **Pinned data (round-3 item 4):** fetch fresh Binance spot DAILY OHLCV (with high and low) for the
  whole ever-listed pool and 1m klines as below, all through **2026-11-06**, from data.binance.vision
  / data-api.binance.vision. The repo's existing daily cache is NOT used (it ends 2026-10-10 and has
  no high/low). Before the first backtest run, freeze a manifest (every file path, size, SHA-256,
  and the data end date) and commit it; any later re-download that changes a hash invalidates the run.
- Binance spot 1m klines. Cheap path: from daily bars, flag coin-days d where
  low_d <= O_d x 0.80 **or** low_{d+1} <= O_d x 0.80 (next-day fills on day-d bids); download 1m for
  each flagged coin-day and for **at least 6 days after each fill** (72h hold + 48h outage + margin).
  Daily bars of all held coins for marking; BTCUSDT daily for the breadth split.
- C4 x2 + PAXG weights: baseline from the `fund.core.decide` replay; combined book from its own path.

## Periods

- **One continuous run** from 2018-08-17 to 2026-10-31, starting at USD 1,000, for each book
  (round-3 item 2). DEV = the 2018-08-17..2024-12-31 slice (effectively 2020 onward); HOLDOUT = the
  2025-01-01..2026-10-31 slice. Each slice's max DD, CAGR and Sharpe are computed within the slice
  (peak reset at the slice start). No fresh start at the HOLDOUT boundary.
- Then forward paper on a separate ledger that also simulates C4 (the live C4 paper ledger is
  untouched).

## Contamination disclosure

- Tier levels (-20/-30/-45%) and the -45% cap were written after seeing the 2025-10-10 magnitudes.
- The scout named DEV events from memory (Mar 2020, May 2021, Aug 2024). DEV daily bars were seen in
  the C4 research (not this rule, not 1m data).
- All counts in "Event counts already seen" were computed by the critics and seen before this
  version was finalised; the G1 thresholds were confirmed after seeing them and were NOT moved.
- DEV is the decision period but not pristine; HOLDOUT is contaminated by 2025-10-10; only the
  forward test is clean.

## Metrics (all reported)

- Episodes: count, mean, median, hit rate, worst, best, time in position; split by tier, by coin, by
  H-cascade vs -20%-only, by market-wide vs single-coin.
- Leave-one-episode-out lower bounds (full list).
- Share of PnL from coins in T20 < 6 months; share of fills in coins Tokocrypto lists today
  (survivor-biased label); share in coins not listed there (e.g. ZEC).
- $5 skips; positions hit by the series-end rule; exit sells delayed by PERCENT_PRICE_BY_SIDE.
- Ladder sleeve: CAGR, max DD, best/worst week, share of weeks >= +40% (owner's ambition, reported,
  not a gate).
- Both books, marked at daily close: CAGR, Sharpe, max DD, Calmar, correlation, best/worst week, share
  of weeks >= +40%, live-halt fire dates.

## Gates (locked; all must pass)

- **G0 Venue precondition (procedure, round-2 item 8).** Owner's action, before any result is used:
  1. Pick a Tokocrypto USDT pair whose Binance spread is wider than one tick at that moment; use a
     less liquid pair if no liquid one qualifies. (No fallback to an existing price level.)
  2. Place a resting limit buy of ~USD 6, with a distinctive non-round quantity, on Tokocrypto at a
     price strictly inside Binance's spread, i.e. a price level that does not exist in the book, and
     record the time.
  3. Poll Binance public depth (`data-api.binance.vision/api/v3/depth`) for 30 s: PASS-1 only if a
     NEW level appears at exactly that price with exactly the order's quantity (round-3 item 6).
  4. Leave it until it fills or 24h: PASS-2 if the Tokocrypto fill time matches a Binance public
     trade (`aggTrades`) at that price within 2 s.
  5. Cancel if unfilled after 24h; repeat once on another pair if PASS-2 could not be observed.
  Pass = PASS-1 and PASS-2. Fail on either = Binance 1m data is not fill evidence; the test is VOID.
- **G1 Minimum counts (DEV, after all rules).** >= 40 H-ladder and >= 20 H-cascade episodes.
  Fewer = "insufficient evidence", not a pass; no rule changes to get more.
- **G2 Significance (DEV).** Each must pass:
  - G2a H-ladder; G2b H-cascade; G2c H-cascade on the market-wide subset (round-2 item 7).
  - Bootstrap (round-2 item 9): blocks = fixed calendar quarters of DEV (all quarters from 2018Q3
    to 2024Q4, including quarters with no episodes). Each draw resamples that many quarters with
    replacement; statistic = pooled mean = (sum of episode returns in drawn quarters) / (number of
    episodes in drawn quarters). A draw with zero episodes is discarded and redrawn (count of
    redraws reported). 10,000 kept draws, seed 20261109. Pass = mean > 0 and the 5th percentile of
    the draws > 0 (one-sided 95%).
  - G2c has no separate minimum count; it is expected to have ~17 episodes over 13 quarters
    (MEASURED daily approximation) and may fail for lack of power. That outcome is accepted in
    advance.
- **G3 No single-episode dependence (DEV).** For G2a, G2b and G2c, the 5th percentile stays > 0
  after dropping the single best episode.
- **G4 Tail test (DEV).** G2a, G2b and G2c must still pass under both stresses at once:
  (i) time-stop exits carry 2% extra slippage, **replacing** (not adding to) the 0.5%; and
  (ii) every episode with >= 10 distinct coins filled has its exits delayed by 48h.
  Mechanics (round-3 item 3):
  - (a) The set of >= 10-coin episodes is taken from the PRIMARY run (expected 19 of 58 H-ladder and
    17 of 26 H-cascade DEV episodes under compounding; 15 of 49 and 14 of 23 at a fixed $1,000;
    MEASURED by the critics) and is not recomputed under stress.
  - (b) Fills, episode membership and episode denominators are held FIXED from the primary run;
    only the exits are repriced. The book is not re-simulated.
  - (c) Outage window per affected episode = [episode's first fill, first fill + 48h]. No exit order
    exists during the window; exits of fills inside it become eligible from the 1m bar after the
    first poll after the window ends. Fills after the window are handled normally. The time-stop
    clock runs from each fill (fill + 72h); if the outage window covers that moment, the time-stop
    exit happens at the first poll after the window ends, at the open of the next 1m bar, minus the
    2% slippage. Because every fill is at or after the episode's first fill, fill + 72h is always
    after first fill + 48h, so this clause cannot trigger; it is kept only as a stated rule. A target
    crossed during the window is filled only by a print after it ends (no retroactive fills).
  - (d) Unaffected episodes use the primary exits except that their time-stop exits also carry the
    2% (replacing 0.5%).
- **G5 Holdout (run twice).** HOLDOUT pooled mean H-ladder episode return > 0 (a) with all episodes
  and (b) on a re-run with no ladder bids for the whole episode that contains any 2025-10-10 fill
  (round-2 item 10). Both must be > 0. H-cascade (3 holdout episodes) reported, not gated.
- **G6 Combined book vs unmodified C4 (DEV, both unhalted).**
  - Significance: paired stationary bootstrap of the annualised Sharpe difference (combined minus
    C4). **New code, spec:** Politis-Romano stationary bootstrap with geometric block lengths, mean
    20 days, over the common daily index; each draw generates ONE index sequence and applies it to
    both daily return series; statistic = Sharpe(combined draw) - Sharpe(C4 draw), annualised with
    sqrt(365); 10,000 draws, seed 20261109. Pass = 5th percentile > 0. Unit test required before use:
    two identical series give a difference of exactly 0 in every draw; a series plus a constant
    positive daily return gives a positive difference in every draw.
  - Materiality hurdle (size, not significance): DEV Sharpe difference >= +0.10 (~+2–2.5%/yr of
    account return, INFERRED by the round-1 critic).
  - Max DD of the combined book no more than 5 percentage points worse than C4's.
- **G7 Combined book (HOLDOUT, unhalted, run twice: with all episodes and without the 2025-10-10
  episode as defined in G5).** Max DD no more than 5pp worse than C4 and CAGR not lower, in both runs.
- **G8 Single-episode tail.** No episode's H-ladder return (all fills) is below -25% of its episode
  denominator, in DEV or HOLDOUT.

## Sensitivities (information only; none can rescue a failed gate)

1. Once-daily exit (no poller): exits only at 00:10 UTC runs; a crossed target fills at the run's 1m
   open; the time stop rounds up to the next run.
2. Random outage: all exits delayed 48h on a random 5% of episodes (fixed seed, 1,000 redraws).
3. Missed runs on a random 5% of days (bids keep resting at their original levels).
4. Late run: the daily run (cancel, C4, bids) happens at 02:30 UTC instead of 00:10 (the one paper
   run so far executed at ~02:27 UTC); fills count from 02:31.
5. Both books under the live sticky 30% halt.
6. Halt/series-end haircut 100%.
7. Tiers shifted +/-5pp; time stop 24h; fees 0.6% per side; start equity USD 10,000.

## Preconditions for real money (beyond passing every gate)

1. G0 passed.
2. **Poller uptime demonstrated in forward paper:** at least 60 consecutive days in which >= 99% of
   scheduled 15-minute polls completed and no gap between completed polls exceeded 2 hours, logged
   by the poller itself and checked by the fund-ops agent.
3. The forward paper ledger (simulating C4 + ladder) has run through the kill rules below without a
   stop for the same period.

## Kill rule (forward paper / later small real)

- Stop if 3 consecutive episodes are net negative, if any single episode loses > 15% of its
  episode denominator, or on the ladder drawdown rule: S_t = K + cumulative realised ladder PnL +
  unrealised PnL of open ladder positions at the daily close, with K = 0.8 x the forward book's
  start equity and realised PnL net of fees. Stop when S_t falls more than 20% below its running peak.
- Stop immediately (infeasible) if Tokocrypto fills stop matching Binance's book, if Tokocrypto halts
  trading or its API during a > 10% market drop while bids rest, or if the poller is down for more
  than 24h during an open episode twice.
- If no market-wide H-cascade episode occurs in 12 months of forward running, report "untested",
  not "working".
