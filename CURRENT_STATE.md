# CURRENT STATE (AUTHORITATIVE, MODEL-FACING)

Last updated: 2026-02-17  
Precedence (if any source conflicts):
1) THIS FILE → 2) Code (executed logic) → 3) Latest run log (.txt) → 4) SYSTEM_OVERVIEW.md → 5) EXPERIMENTS_LOG.md → 6) GLOSSARY_AND_GOTCHAS.md

---

## A) Latest completed experiment & baseline

Experiment: **EXPERIMENT-013**  
Run ID: **Final-Train-Test_20260216_203124**  
Reward function hash: **09f8bc66**  

Change vs EXP-012 (implemented & executed):
- Added **stop-loss cooldown gate**: mask new entries after **K=4** stop-loss exits in the same day (mechanics/action-masking only).
- Added diagnostics (TXT-only): weekday performance table, session-flag sanity check, cooldown summary, and Train reward component aggregates.

### Train (2021-01-04 → 2024-12-31; 1035 days)
| Metric | Value |
|---|---|
| Trades/day | 9.2 |
| Cap-hit days | 756/1035 |
| Total PnL | $9,065.34 |
| Avg daily PnL | $8.76 |
| Profit factor | 1.16 |
| Sharpe (ann.) | 0.95 |
| Win rate (daily) | 48.3% |
| Max intraday DD | -$276.72 |
| Max DD from cum. peak | $3,196.61 |
| Daily DD breaches (5%) | 0/1035 |
| Best day | $679.02 |
| Worst day | -$276.72 |
| Max consec. wins | 8 |
| Max consec. losses | 8 |

### Test (2025-01-02 → 2025-12-31; 259 days)
| Metric | Value |
|---|---|
| Trades/day | 9.3 |
| Cap-hit days | 192/259 |
| Total PnL | $8,915.79 |
| Avg daily PnL | $34.42 |
| Profit factor | 1.68 |
| Sharpe (ann.) | 3.24 |
| Win rate (daily) | 53.3% |
| Max intraday DD | -$291.41 |
| Max DD from cum. peak | $1,253.57 |
| Daily DD breaches (5%) | 0/259 |
| Best day | $601.53 |
| Worst day | -$291.41 |
| Max consec. wins | 9 |
| Max consec. losses | 5 |

### EXP-013: Calendar-month DD (Test 2025, evaluation-only; correct calendar-month peak DD)
- **Breach observed**: **2025-09 Peak-DD = -$1,223.57** (> $1,000 FTMO monthly DD limit)
- Other months peak-DD were ≤ -$887.98 (Oct) and ≤ -$843.14 (Nov), but still below $1,000.

### EXP-013: Correctness / integrity diagnostics (Test 2025)
- ATR fallback usage: 10 / 345,971 bars (0.003%)
- Stop-loss integrity: stop_loss exits with positive PnL = 0 / 686 (0.00%)
- Exit price sources: bar_close 897; bar_open 825; stop_price 686
- Action distribution: HOLD 94.0%; OPEN_LONG 2.1%; OPEN_SHORT 1.2%; CLOSE 2.7%
- Blocked actions: blocked reversals (min-hold) 8,228; blocked manual closes (min-hold) 7,941
- Masked opens: budget 0; cooldown 0 (note: cooldown triggered days reported but masked-open count likely not capturing “masked attempts”)
- Trade duration (Test): p50 15; mean 16.0; stop_loss mean 9.9; manual/reversal mean ~18.5/18.4

### EXP-013: Session sanity & weekday (Test 2025)
- Session-flag sanity (bars >0.5): Asia 26.4%; London 31.3%; NY 42.3%; overlaps 0%
- Weekday performance: Friday best ($56.28/day, PF 2.29); Monday weakest ($13.66/day, PF 1.25)

Key observations from EXP-013:
- ✅ Profitability recovered vs EXP-012 (Test $34.42/day vs $12.94/day).
- ✅ Sharpe improved materially (Test 3.24).
- ✅ Daily DD compliance preserved (0 breaches).
- ✅ Session flags are non-degenerate (sanity check passes).
- ❌ Calendar-month DD still fails FTMO monthly limit due to Sep 2025 breach (-$1,223.57).
- ⚠️ Session slice “trades by session” is still suspicious/likely wrong (shows almost all trades in Asia despite NY bars being present). Must fix diagnostics before session-aware mechanics.

---

## B) Current experiment
Status: **EXPERIMENT-014 – Implemented (pending execution/results)**

**Primary hypothesis (coherent)**: The remaining FTMO monthly DD breach is driven by stop-loss cluster days; additionally, the daily 10-entry cap is likely being consumed too early in the day, preventing participation in later sessions (London/NY). A **session-aware entry budget allocation** (mechanical masking, no reward changes) plus a **two-stage risk-off gate based on stop-loss streaks** and **correct entries-by-session + cap-timing diagnostics** will:
1) distribute trades across Asia/London/NY,
2) reduce tail-risk days that cause monthly DD breaches,
3) preserve daily DD compliance and maintain ≥7 trades/day.

**Planned changes (EXPERIMENT-014)**:
1. **Session-aware entry budget allocation (mechanics + action masking)**:
   - Reserve a minimum number of entries for London and NY (small, e.g., 2 each), preventing Asia from consuming full daily cap before later sessions.
   - Implement purely via action masking for OPEN_LONG/OPEN_SHORT; no reward term.

2. **Two-stage stop-loss risk-off gate (mechanics + action masking)**:
   - Soft risk-off after 2 stop-loss exits/day (reduced allowed opens until session transitions).
   - Keep hard cooldown at 4 stop-loss exits/day (existing EXP-013 behavior), but ensure it has measurable effect.

3. **Diagnostics (TXT-only): entries-by-session + cap timing + cooldown mask effectiveness**:
   - Entries-by-session table (Train + Test): entry counts and PnL contribution by session.
   - Cap timing: distribution of the 10th entry timing (session and bar index/timestamp).
   - Cooldown effectiveness: count bars where cooldown mask active and count OPEN attempts while masked.
   - Replace/fix the old session slice summary to use consistent entry attribution.

**Success criteria (EXPERIMENT-014)**:
- Must not regress:
  - Daily DD breaches remain 0 (Train/Test)
  - Trades/day (Test) remains ≥ 7.0
  - Stop-loss integrity violations remain 0
- Session participation target:
  - London + NY combined entries ≥ 40% of all entries (initial target; adjust after first diagnostic run if needed)
- Risk target:
  - Calendar-month peak DD ≤ $1,000 for all Test months (eliminate Sep breach)
- Profitability target:
  - Maintain Test avg daily PnL ≥ $30, aim ≥ $50 without DD regression.

---

## C) Invariants (do not change without explicit decision)

**Episode + state:**
- Episode = one filtered trading day; must have ≥ 60 bars
- State = (60, 29) features-only
- Internal env day array = (n_bars, 32) = OHLC(4) + 29 features

**Mechanics:**
- Actions: 4 (HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION)
- Execution: decide at t, advance to t+1, execute at next bar open/close
- Risk sizing: 0.5% of current equity at entry; lot clipped [0.01, 10.0]
- Position sizing formula: unchanged
- PnL calculation: pip-based formula unchanged
- Min hold enforced: 15 bars (hard override to HOLD if close attempted earlier; enforced on reversals since EXP-011)
- Max duration: 90 bars
- EOD forced close
- Reward clip: [-5, +5]
- Daily DD penalty triggers once/day if drawdown from INITIAL_BALANCE exceeds 5%

**IMMUTABLE (since EXP-012):**
- ATR_14 must be resolved by feature name via feature_index_map (no hard-coded indices).
- ATR_14 fallback rule: if ATR_14 < 1.0 or non-finite, use ATR_14 = 4.0 for stop-loss and any ATR-dependent computations.
- All session/volatility diagnostics must use feature-name mapping, not hard-coded indices.
- Stop-loss integrity instrumentation must remain (exit_price_source + positive-PnL checks).

**Phase C windows:**
- Train: 2021-01-04 → 2024-12-31
- Test: 2025-01-02 → 2025-12-31

---

## D) Active constraints / knobs (current values; EXP-013 baseline)

| Parameter | Value | Notes |
|---|---|---|
| MAX_TRADES_PER_DAY | 10 | Entries only; closes always allowed |
| Action masking | OPEN masked when trades ≥ 10 OR cooldown active | Hard cap + cooldown gate (EXP-013) |
| BUDGET_LAMBDA | 0.0 | Convex budget penalty disabled |
| MIN_TRADES_TARGET | 5 | Soft floor, end-of-day penalty |
| UNUSED_LAMBDA | 0.2 | Penalty coeff for under-trading |
| EXECUTION_PENALTY_R | 0.02 | Per open/close event, in R units |
| ACTION_PENALTY_R | 0.01 | Per non-HOLD action × exposure_scale |
| STOPLOSS_EVENT_PENALTY_R | 0.05 | Per stop-loss exit, in R units |
| REVERSAL_EVENT_PENALTY_R | 0.03 | Per reversal exit, in R units |
| STOPLOSS_COOLDOWN_THRESHOLD | 4 | Mask new entries after K stop-loss exits in same day |
| ATR_MULTIPLIER | 3.0 | EXP-009 |
| MANUAL_EXIT_AT_OPEN | True | EXP-009 |
| INACTIVITY_BASELINE_REWARD | 0.00025 | EXP-010 |
| PATIENCE_REWARD | 0.004 | EXP-010 |
| MIN_HOLD_BARS | 15 | Enforced on reversals since EXP-011 |
| Eval mode | argmax | Deterministic |
| n_quantiles | 51 | QR-DQN |
| risk_fraction | 0.25 | Risk-averse action selection |
| quantile_huber_kappa | 1.0 | QR-DQN |
| CQL enabled | False | Disabled |

---

## E) Gaps to FTMO target (based on latest completed experiment: EXP-013)

**Primary gap – monthly DD compliance:**
- Test calendar-month peak DD breached FTMO monthly limit in:
  - **2025-09 Peak-DD -$1,223.57**

**Primary gap – profitability:**
- Test avg daily PnL = **$34.42** (target $50–$100) → needs ~1.5× to reach minimum target band.

**Secondary gap – intraday tail:**
- Worst intraday DD day **-$291.41** (safe vs -$500 but above desired ~-$200).

**Open gap – session participation confirmation:**
- Session flags are sane, but trade-by-session diagnostics are likely incorrect; need correct entries-by-session and cap-timing logs before adjusting session-aware controls.

---

## F) Next Experiment Hypotheses (ranked)
1) **Session-aware budget allocation + risk-off gating (EXP-014 planned):** enforce mechanical distribution of limited entries across sessions and reduce stop-loss cluster cascades in peak-loss days.
2) **Tail-risk focus:** reduce frequency of 4-stop-loss days; these dominate worst days and likely drive monthly DD breach.
3) **Diagnostics-first on session behavior:** verify actual entry distribution and “cap timing” before any reward changes.

---

## G) Parking Lot (not current priority)
- PER only after session allocation + tail gating stabilizes monthly DD.
- Reward coefficient retuning only after session distribution and tail behavior are mechanically stabilized and well-measured.
- Risk_fraction / distributional objective changes only after isolating regime failures with new diagnostics.

---

# APPEND (2026-02-17): EXPERIMENT-014 COMPLETED + GOALS UPDATE

## A2) Latest completed experiment (supersedes baseline for next iteration)

Experiment: **EXPERIMENT-014**  
Run ID: **Final-Train-Test_20260217_124508**  
Reward function hash: **09f8bc66**  
Mechanics hash: **bea1c04b**  

### Train (2021-01-04 → 2024-12-31; 1035 days)
| Metric | Value |
|---|---|
| Trades/day | 7.3 |
| Total trades | 7,564 |
| Total PnL | $6,923.59 |
| Avg daily PnL | $6.69 |
| Profit factor | 1.14 |
| Sharpe (ann.) | 0.78 |
| Win rate (daily) | 45.2% |
| Max intraday DD (single day) | -$233.54 |
| Max DD from cum. peak | $2,168.62 |
| Best day | $842.95 |
| Worst day | -$233.54 |

### Test (2025-01-02 → 2025-12-31; 259 days)
| Metric | Value |
|---|---|
| Trades/day | 6.8 |
| Total trades | 1,757 |
| Total PnL | -$1,482.00 |
| Avg daily PnL | -$5.72 |
| Profit factor | 0.90 |
| Sharpe (ann.) | -0.66 |
| Win rate (daily) | 42.1% |
| Max intraday DD (single day) | -$248.22 |
| Max DD from cum. peak | $2,315.04 |
| Best day | $801.01 |
| Worst day | -$248.22 |

### EXP-014: Calendar-month DD (Test 2025, evaluation-only; calendar-month peak DD)
- Monthly DD breaches (≤ -$1,000 limit violated):
  - 2025-06 Peak-DD = -$1,509.33
  - 2025-11 Peak-DD = -$1,199.92

### EXP-014: Session participation (Test 2025, by trade_record['entry_session'])
- Asia: 44.2% of entries, PnL sum -$675.93
- London: 32.4% of entries, PnL sum -$202.33
- NY: 23.4% of entries, PnL sum -$603.74
- London+NY combined: 55.8% of all entries (session allocation achieved)

Key observations:
- ✅ Daily DD compliance remains strong (max intraday DD -$248 < -$500).
- ✅ Session participation target achieved mechanically.
- ❌ Test edge is negative (PF 0.90, Sharpe -0.66, total PnL -$1,482).
- ❌ Monthly DD breaches persist (Jun, Nov).

## B2) Next experiment
Status: **EXPERIMENT-015 – Pending**

Primary hypothesis:
- The agent needs explicit, *minimal* preference for **high R-multiple outcomes (≥2R)** and a **more direct stop-loss cool-off gate** to reduce stop-loss cluster cascades. This should improve calendar-month DD and profitability without forcing trade quantity.

Planned changes (EXPERIMENT-015):
1) Reward shaping (R-normalized, minimal): add a bounded bonus for realized R ≥ 2.0 at trade close; optionally suppress tiny wins to reduce insignificant trading.
2) Mechanics: replace soft gate with a fixed cool-off window after 2 stop-loss exits (mask OPEN for N bars).
3) Diagnostics: add R-multiple distribution tables and gate-attempt integrity counters to TXT logs.

Success criteria (EXPERIMENT-015):
- Risk: calendar-month peak DD ≤ $1,000 for all Test months.
- Profit: target **≥ 10% per month** (evaluation framing), while keeping daily DD compliance.
- Behavior: trade frequency may decline; not required to trade every day; focus on positive expectancy.

## E2) Updated goals (AUTHORITATIVE)
- Primary profit goal: **≈ 10% per calendar month** on $10k (≈ $1,000/month).  
- Risk limits (unchanged):
  - Daily DD limit: ≤ 5% of $10k ($500) (in-env penalty/constraint)
  - Monthly DD limit: ≤ 10% of $10k ($1,000) (evaluation-only; must be calendar-month peak-based)
- Trade frequency: no minimum required; up to 10 entries/day allowed; prioritize edge over activity.