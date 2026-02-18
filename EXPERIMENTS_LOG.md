# EXPERIMENTS LOG (APPEND-ONLY, MODEL-FACING)

Rules:
- Never edit completed experiments except to add missing run metadata.
- Always append new experiments at the bottom.
- Each experiment should change as few things as possible.
- Format: hypothesis â†’ change â†’ results â†’ decision/learnings

---

## QUICK COMPARISON TABLE (most recent Phase C runs)
| Experiment | Run ID | Test Avg $/day | Test PF | Test Sharpe | Test Trades/day | Test Cap-hit | Test Max Intraday DD | Test Max DD Peak | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EXP-005 | Final-Train-Test_20260205_120614 | 25.47 | 1.32 | 1.83 | 9.9 | 94.6% | -488.90 | 1,530.64 | R-normalized execution + action penalties; epsilon decay mismatch (ended ~0.32) |
| EXP-006 | Final-Train-Test_20260206_222712 | 6.75 | 1.37 | 0.82 | 3.2 | 12.7% | -405.06 | â€“ | Epsilon fix worked (Îµâ†’0.01) but thresholded EOD cluster penalty caused collapse |
| EXP-007 | â€“ | â€“ | â€“ | â€“ | â€“ | â€“ | â€“ | â€“ | Smooth stop-loss event penalty; results pending |
| EXP-008 | Final-Train-Test_20260207_215022 | 25.43 | 1.34 | 1.81 | 9.4 | 72.2% | -457.34 | 2,001.28 | QR-DQN + risk-averse action selection; CQL disabled; performance ~flat vs EXP-005 |
| EXP-009 | Final-Train-Test_20260210_215055 | 17.89 | 1.86 | 2.77 | 6.0 | 26.6% | -236.12 | 743.38 | ATRÃ—3 + manual exit at next bar OPEN + TXT diagnostics. Major DD improvement + correct calendar-month DD; profitability regressed due to under-trading / reward dominance (inactivity + early-exit penalty). |
| EXP-010 | Final-Train-Test_20260211_120041 | 42.73 | 2.64 | 4.58 | 9.6 | 87.6% | -265.66 | 481.85 | Reward rebalance fixed EXP-009 under-trading; DD compliance remains strong. New concern: median duration 1 bar; stop_loss 57.8% and reversals 35.9% suggest churn/flip behavior. |
| EXP-011 | Final-Train-Test_20260211_203820 | 39.04 | 2.33 | 3.81 | 10.0 | 99.2% | -380.99 | 654.52 | Anti-churn: min-hold enforced on reversals + reversal event penalty. Compliance strong but profitability below EXP-010; median duration still 1 bar due to stop_loss-heavy micro-trades. Diagnostics added but regime slices currently broken; possible stop_loss labeling/logging inconsistency (profitable stop_loss trades in CSV). |
| EXP-012 | Final-Train-Test_20260216_133117 | 12.94 | 1.24 | 1.41 | 10.0 | 100.0% | -351.30 | 2,561.82 | Correctness + diagnostics: ATR_14 by name + ATR fallback + stop-loss integrity instrumentation + fixed diagnostics. Stop_loss positive PnL violations: 0. Major regression in profitability and calendar-month DD (Feb peak-DD -1089; Jun peak-DD -2075). |
| EXP-013 | Final-Train-Test_20260216_203124 | 34.42 | 1.68 | 3.24 | 9.3 | 74.1% | -291.41 | 1,253.57 | Stop-loss cooldown gate (K=4) + weekday/session sanity + Train reward aggregates. Calendar-month DD still breaches in Sep (peak-DD -1,223.57). |

Last updated: 2026-02-16 (append-only update)

APPEND (2026-02-10):
- Clarification: "Test Max DD Peak" in the quick table currently reflects whole-period cumulative peak-to-trough drawdown, not calendar-month segmented drawdown. EXP-009 will add calendar-month DD reporting in TXT logs.
- EXPERIMENT-009 implemented: ATR_MULTIPLIER=3.0, manual exit at next bar OPEN, TXT diagnostics (calendar-month DD, top-10 worst days, reward components). Results pending.

APPEND (2026-02-11):
- EXPERIMENT-009 completed (Run ID: Final-Train-Test_20260210_215055, reward hash: 7c07eff1).
  - Key outcome: Test risk/compliance improved substantially (max intraday DD -236; whole-period peak DD 743; calendar-month peak DD all < 412),
    but Test avg daily PnL regressed to $17.89 with trades/day 6.0 and cap-hit days 69/259.
  - Diagnostics show reward dominance issues: early_exit_penalty dominates negative reward and inactivity_baseline dominates positive reward, suggesting misaligned incentives and under-trading.
- EXPERIMENT-010 implemented (pending results): Reward rebalancing to address EXP-009 profitability regression.
  - Reduced INACTIVITY_BASELINE_REWARD 10Ã— (0.0025 â†’ 0.00025) and PATIENCE_REWARD 5Ã— (0.02 â†’ 0.004).
  - Restricted early_exit_penalty to manual closes only (not stop-outs/reversals).
  - Added TXT diagnostics: trades/day histogram, trade duration percentiles by exit_reason, trade-level win rate.

APPEND (2026-02-11, later):
- EXPERIMENT-010 completed (Run ID: Final-Train-Test_20260211_120041, reward hash: e27569e3).
  - Test improved strongly: avg daily PnL $42.73, PF 2.64, Sharpe 4.58, trades/day 9.6, cap-hit 227/259 (87.6%).
  - Risk/compliance remained strong: daily DD breaches 0/259; max intraday DD -265.66; calendar-month peak DD worst -473.30 (all months far < $1,000).
  - Behavior diagnostics surfaced a new potential pathology: median trade duration 1 bar; stop_loss exits 57.8% and reversals 35.9% indicate churn/flip behavior likely bypassing MIN_HOLD_BARS intent via reversal logic.
  - Next step should focus on reducing reversal-driven churn without reintroducing under-trading or DD regression.

APPEND (2026-02-11, final):
- EXPERIMENT-011 implemented (pending results): Anti-churn mechanics and diagnostics.
  - Enforces MIN_HOLD_BARS (15 bars) on reversals: OPEN_* while in position now blocked if duration < 15, treated as HOLD. Closes min-hold constraint loophole.
  - Adds REVERSAL_EVENT_PENALTY_R = 0.03 (R-normalized) when exit_reason == "reversal" to lightly discourage reversal churn.
  - Adds TXT diagnostics: stop-loss intensity histogram (SL exits/day buckets), volatility quintile table (using mean ATR), session slice summary (Asia/London/NY).
  - Hypothesis: enforcing min-hold on reversals + light reversal penalty will increase median trade duration, reduce stop-loss/reversal frequency, and lift Test avg daily PnL to â‰¥ $50/day without DD regression.

APPEND (2026-02-12):
- EXPERIMENT-011 completed (Run ID: Final-Train-Test_20260211_203820, reward hash: 09f8bc66).
  - Test results: avg daily PnL $39.04, PF 2.33, Sharpe 3.81, trades/day 10.0, cap-hit 257/259 (99.2%).
  - Compliance remained strong: daily DD breaches 0/259; max intraday DD -380.99; calendar-month peak DD worst -654.52 (all months < $1,000).
  - Behavior: reversal min-hold enforcement appears effective (reversal durations centered ~20 bars), but median duration remains 1 bar due to stop_loss-heavy micro-trades (~82% of trades exit as stop_loss).
  - Diagnostics note: stop-loss intensity + volatility quintile + session slice reporting currently appears incorrect (NaNs / zeros / Unknown sessions) due to date/mapping issues.
  - Correctness concern flagged: exported trade CSV shows many trades labeled stop_loss with positive PnL, indicating likely exit_reason labeling or stop execution/logging inconsistency. Next experiment should prioritize mechanics/logging integrity before further reward tuning.

---
## EXPERIMENT-000 â€“ Baseline Phase C Final-Train-Test
Date: 2026-02-01
Goal: Establish baseline, identify FTMO gaps.

Test (2025) results:
- Trades/day: 67.3 | Avg daily PnL: $63.44
- PF: 1.35 | Sharpe: 1.89
- Max intraday DD: -$1,267 | Max DD from peak: ~$5,999

Decision: Not FTMO-compliant â€“ extreme overtrading (67 trades/day). Need hard trade cap.

---

## EXPERIMENT-001 â€“ Deterministic evaluation (argmax default)
Date: 2026-02-03
Change: Evaluation uses deterministic argmax by default; softmax preserved as optional.
Run type: Code change only (no new run).

Decision: Standardize on argmax evaluation for comparability across experiments.

---

## EXPERIMENT-002 â€“ Hard daily entry cap + convex budget penalty
Date: 2026-02-03
Run ID: Final-Train-Test_20260203_213703

Hypothesis: Hard cap + convex penalty will reduce overtrading to FTMO-compliant levels.

Changes vs EXP-001:
- MAX_TRADES_PER_DAY = 10 (mask OPEN actions after cap)
- Convex entry penalty: -BUDGET_LAMBDA Ã— (budget_usageÂ²), BUDGET_LAMBDA = 0.5

Results:
| Split | Trades/day | Cap-hit days | Avg daily PnL | PF |
|---|---|---|---|---|
| Train | 10.0 | 1035/1035 | $13.64 | â€“ |
| Test | 4.1 | 0/259 | $3.61 | â€“ |

Decision: Cap worked mechanically. Convex penalty over-suppressed profitability on Test (only $3.61/day). Disable convex penalty, try soft floor instead.

---

## EXPERIMENT-003 â€“ Disable convex penalty; add unused-budget penalty (soft floor)
Date: 2026-02-04
Run ID: Final-Train-Test_20260204_120420

Hypothesis: Removing convex penalty recovers profitability; soft floor ensures minimum engagement.

Changes vs EXP-002:
- BUDGET_LAMBDA = 0.0 (disabled)
- EOD unused-budget penalty: MIN_TRADES_TARGET = 3, UNUSED_LAMBDA = 0.2

Results:
| Split | Trades/day | Cap-hit days | Avg daily PnL | PF |
|---|---|---|---|---|
| Train | 10.0 | 1035/1035 | $23.09 | â€“ |
| Test | 3.7 | 0/259 | $14.10 | 1.30 |

Decision: Profitability recovered vs EXP-002. Test still far below FTMO target. Execution penalty may be too heavy â€“ try reducing it.

---

## EXPERIMENT-004 â€“ Reduce execution friction
Date: 2026-02-04
Run ID: Final-Train-Test_20260204_211923

Hypothesis: Halving execution penalty reduces friction and allows more profitable trades through.

Changes vs EXP-003:
- Execution penalty reduced 50%: -$1.50 â†’ -$0.75 per open/close event
- MIN_TRADES_TARGET raised from 3 â†’ 5

Results:
| Split | Trades/day | Cap-hit days | Avg daily PnL | PF | Sharpe | Max intra DD | Max DD peak |
|---|---|---|---|---|---|---|---|
| Train | 10.0 | 1035/1035 | $19.76 | 1.32 | 1.68 | -$476.09 | $3,695.08 |
| Test | 4.5 | 2/259 (0.8%) | $14.45 | 1.28 | 1.41 | -$347.09 | $1,091.14 |

Decision: Small Test improvement. Train/Test behavior gap persists (cap saturation 100% vs 0.8%). Penalties mix R-based trade rewards with absolute dollar penalties â†’ incoherent scaling. Try R-normalizing all penalties.

---

## EXPERIMENT-005 â€“ R-normalize execution + action penalties
Date: 2026-02-05
Run ID: Final-Train-Test_20260205_120614
Reward function hash: 51a41cf4

Hypothesis: Converting dollar-based penalties to R-units restores coherent reward scaling, closes Train/Test behavior gap, and improves Test PnL.

Changes vs EXP-004:
- Execution penalty: -EXECUTION_PENALTY_R = -0.02 R per open/close event (was -$0.75)
- Action penalty: -ACTION_PENALTY_R = -0.01 R Ã— exposure_scale per non-HOLD action (was -$0.10 Ã— exposure_scale)

Results:
| Split | Trades/day | Cap-hit days | Avg daily PnL | PF | Sharpe | Max intra DD | Max DD peak | DD breaches |
|---|---|---|---|---|---|---|---|---|
| Train | 10.0 | 1035/1035 (100%) | $26.67 | 1.44 | 2.21 | -$445.83 | $2,908.07 | 0/1035 |
| Test | 9.9 | 245/259 (94.6%) | $25.47 | 1.32 | 1.83 | -$488.90 | $1,530.64 | 0/259 |

Key findings:
1. âœ… Train/Test behavior gap closed â€“ Test now hits cap 94.6% of days (was 0.8%)
2. âœ… Test PnL nearly doubled ($14.45 â†’ $25.47)
3. âœ… Sharpe improved significantly (1.41 â†’ 1.83)
4. âš ï¸ Max intraday DD worsened (-$347 â†’ -$489), approaching 5% threshold
5. âš ï¸ Max DD from peak worsened ($1,091 â†’ $1,531), exceeds 10% FTMO monthly DD target
6. âš ï¸ Trade-level win rate only 40.64% â€“ relies on win/loss asymmetry
7. âŒ Still far from FTMO profit target ($25.47 vs $50-100/day)

Epsilon at end of training: 0.3202 (did NOT reach minimum â€“ only 326,323 training steps vs 1,080,121 decay steps)

Decision: R-normalization was correct direction. Next priorities: (1) improve per-trade edge (win rate or avg win size), (2) contain max DD, (3) investigate why epsilon didn't decay fully.

---

## EXPERIMENT-006 â€“ Fix epsilon decay accounting + add stop-loss cluster penalty
Date: 2026-02-06
Goal: Correct epsilon decay to use env-steps (not update-steps) so epsilon reaches minimum by ~85% of training. Add EOD stop-loss-cluster penalty to discourage persistent trading into regimes that hit stop-loss repeatedly.

Changes vs EXP-005:
1. **Epsilon decay fix**: DQNAgent now tracks `env_steps` and `update_steps` separately. Epsilon schedule uses env_steps.
2. **Stop-loss cluster penalty**: New EOD penalty if stop-loss exits exceed threshold.

Results:
Run ID: Final-Train-Test_20260206_222712

| Split | Trades/day | Cap-hit days | Avg daily PnL | PF | Sharpe | Max intra DD |
|---|---|---|---|---|---|---|
| Train | 10.0 | 1035/1035 (100%) | $19.50 | 1.32 | 1.62 | -$497.01 |
| Test | 3.2 | 33/259 (12.7%) | $6.75 | 1.37 | 0.82 | -$405.06 |

Key findings:
1. âœ… Epsilon decay fix worked - epsilon reached 0.01 by end of training
2. âŒ Test performance collapsed vs EXP-005
3. âŒ Likely cause: thresholded EOD cluster penalty created degenerate avoidance behavior

Decision: Preserve epsilon fix. Replace thresholded cluster penalty with smooth per-event penalty (EXPERIMENT-007).

---

## EXPERIMENT-007 â€“ Smooth stop-loss event penalty (remove thresholded cluster penalty)
Date: 2026-02-07
Goal: Keep epsilon accounting fix from EXP-006, replace thresholded cluster penalty with smooth per-event penalty.

Changes vs EXP-006:
1. Remove stop-loss cluster penalty and counters.
2. Add STOPLOSS_EVENT_PENALTY_R = 0.05 applied immediately on stop-loss exits.
3. Preserve epsilon fix.

Results: **Pending**
---

## EXPERIMENT-008 â€“ Distributional QR-DQN + Risk-Averse Action Selection
Date: 2026-02-07
Run ID: Final-Train-Test_20260207_215022
Reward function hash: 735b8c2f

Goal: Implement distributional RL with quantile regression and risk-averse action selection to improve regime robustness and trade selectivity.

Changes vs EXP-007:
1. QR-DQN network (51 quantiles/action) using quantile Huber loss.
2. Risk-averse action selection: mean of bottom 25% quantiles (k=13).
3. Disable CQL to isolate QR-DQN effects.
4. Fix STOPLOSS_CLUSTER reporting crash; expand reward hash.

Results:
| Split | Trades/day | Cap-hit days | Avg daily PnL | PF | Sharpe | Max intra DD | Max DD peak |
|---|---|---|---|---|---|---|---|
| Train | 10.0 | 1028/1035 (99.3%) | $23.87 | 1.37 | 1.91 | -$488.90 | $2,363.76 |
| Test | 9.4 | 187/259 (72.2%) | $25.43 | 1.34 | 1.81 | -$457.34 | $2,001.28 |

Decision (preliminary):
- Risk-averse QR-DQN did not materially improve avg daily PnL vs EXP-005, but reduced worst-day intraday DD modestly.
- Peak-to-trough drawdown remains too high; next work should focus on reducing stop-loss churn / tail losses and improving exit mechanics, and add logging for calendar-month DD and reward dominance analysis.
---
## EXPERIMENT-012 – ATR/Stop-loss correctness + diagnostic fixes
Date: 2026-02-16
Run ID: Final-Train-Test_20260216_133117
Reward function hash: 09f8bc66 (unchanged from EXP-011)

Hypothesis: Current performance is limited by (1) mechanics/indexing correctness bugs (ATR_14 column + stop-loss integrity) and (2) broken diagnostics. Fixing ATR_14 resolution by name, adding ATR fallback (< 1.0 → 4.0), and adding stop-loss integrity instrumentation will make results trustworthy and enable safe next-step tuning.

Changes vs EXP-011:
1. Feature-name column mapping: ATR_14, ATR_norm, and session columns resolved by name via feature_index_map (no hard-coded indices).
2. ATR fallback rule: If ATR_14 < 1.0 or non-finite, use ATR_14 = 4.0 for stop-loss distance and lot sizing.
3. Stop-loss integrity instrumentation: Added trade record provenance (exit_price_source, atr_used, atr_was_fallback, stop_distance), track stoploss_positive_pnl_count, print TXT warnings for stop_loss exits with positive PnL.
4. Fixed diagnostics (TXT-only): date keys for correct joins; stop-loss intensity; volatility quintiles via ATR_norm; session slice via session_* columns; action distribution + blocked-action counters.
5. ATR fallback tracking: Count and report ATR fallback usage rate.
6. Exit price source tracking: Count exits by source (stop_price, bar_open, bar_close).

Results (Phase C Final Train/Test):
Train (2021-2024):
- Avg daily PnL: -$2.21 | PF 0.96 | Sharpe -0.24 | Trades/day 10.0 | Daily DD breaches 0/1035
Test (2025):
- Avg daily PnL: $12.94 | PF 1.24 | Sharpe 1.41 | Trades/day 10.0 | Daily DD breaches 0/259
Calendar-month DD (Test): breached FTMO monthly limit in at least Feb (peak-DD -1089.74) and Jun (peak-DD -2075.23).
Stop-loss integrity: stop_loss exits with positive PnL = 0 (827 stop exits).

Decision:
- Correctness baseline established (stop-loss labeling/integrity fixed; ATR mapping fixed).
- Performance regressed materially vs EXP-011; prior edge likely relied on incorrect ATR/indexing.
- Next experiment should focus on reducing sustained loss regimes (monthly DD breaches) via minimal, coherent risk-off gating and improved slice diagnostics, without changing immutables.

---
## EXPERIMENT-013 – Stop-loss cooldown gate + weekday/session diagnostics
Date: 2026-02-16
Status: Implemented (code changes complete, pending execution)
Reward function hash: 09f8bc66 (unchanged from EXP-012)

Hypothesis: With ATR/stop-loss correctness restored, the main driver of FTMO monthly DD breaches is sustained negative-expectancy trading during stop-loss clusters. Adding a minimal, mechanical **stop-loss cooldown gate** (mask new entries after K stop-outs in a day) plus targeted TXT diagnostics (weekday + session-flag sanity + Train reward aggregates) will reduce monthly DD and improve PF/Sharpe without changing immutables or relying on speculative reward retuning.

Changes vs EXP-012:
1. **Stop-loss cooldown gate (mechanics + action masking)**:
   - Added STOPLOSS_COOLDOWN_THRESHOLD = 4 constant in TradingEnvironment.
   - Track stoploss_exits_today counter; incremented in _close_position() when reason == "stop_loss".
   - Modified get_action_mask() to mask OPEN_LONG/OPEN_SHORT when stoploss_exits_today >= STOPLOSS_COOLDOWN_THRESHOLD.
   - Modified step() to track masked_open_cooldown_count separately from budget masking.
   - Added cooldown_triggered boolean flag per episode for diagnostics.
   - **No reward term added** – this is a pure mechanics/action-mask constraint.

2. **Diagnostics: weekday + session-flag sanity (TXT-only)**:
   - Added Test 2025 weekday performance table: days, total/avg PnL, profit factor, worst day, worst intraday DD, avg stop-loss exits/day.
   - Added session-flag sanity check: % bars with session_asia/london/ny > 0.5, overlap rates (asia&london, london&ny, asia&ny, all3).
   - Added cooldown summary: days where cooldown triggered, avg PnL on triggered vs non-triggered days, avg stop-loss exits on triggered days, total masked opens (cooldown).

3. **Diagnostics: Train reward component aggregates (TXT-only)**:
   - Modified train_episode() to accumulate reward_components_sum from info['reward_components'] (matching evaluate_episode_with_export pattern).
   - Added Train 2021-2024 reward components table (Sum and Mean/Episode).
   - Added Train vs Test reward components side-by-side comparison (Mean/Episode) for all shared keys.

Success criteria:
- Must not regress: Daily DD breaches remain 0 (Train/Test); trades/day (Test) remains ≥ 7.0 (allow some reduction due to cooldown); stop-loss integrity violations remain 0 (or ≤ 1 with explicit explanation).
- Risk target: Calendar-month peak DD ≤ $1,000 for all Test months (or at minimum eliminate extreme month like -$2,075).
- Profitability target: Recover Test avg daily PnL to ≥ $25 (minimum), aim ≥ $35 without DD regression.

Results: **Pending execution**

APPEND (2026-02-17):
- EXPERIMENT-013 completed (Run ID: Final-Train-Test_20260216_203124, reward hash: 09f8bc66).
  - Test improved materially vs EXP-012: avg daily PnL $34.42, PF 1.68, Sharpe 3.24, trades/day 9.3, daily DD breaches 0/259.
  - Calendar-month DD (Test) still fails FTMO monthly limit due to Sep 2025 peak-DD -$1,223.57.
  - Session-flag sanity check indicates Asia/London/NY features are non-degenerate, non-overlapping, and cover all bars; however, the trade-by-session slice summary still appears inconsistent and likely requires diagnostic fix before session-aware tuning.

APPEND (2026-02-17 — EXP-014 planned):
- EXPERIMENT-014 planned: Session-aware entry budget allocation + two-stage stop-loss risk-off gate + entries-by-session diagnostics.
  - Addresses remaining FTMO monthly DD breach (Sep 2025 peak-DD -$1,223.57) and confirmed wrong session-slice diagnostics.
  - Key mechanical changes (action masking only, no reward changes):
    1. Session-aware budget allocation: RESERVED_ENTRIES_LONDON=3, RESERVED_ENTRIES_NY=4 → Asia hard limit = 3 entries/day; London hard limit = 6 entries/day; NY uses full cap.
    2. Two-stage stop-loss gate: Soft gate at 2 SL exits/day (1 entry allowed per session until session changes); hard gate at 4 SL exits/day (existing EXP-013 cooldown).
  - Diagnostics additions (TXT-only):
    3. Entries-by-session table using trade_record['entry_session'] (set at open execution, not re-inferred from bar data).
    4. Cap timing: distribution of 10th-entry bar index / session across days.
    5. Gating effectiveness: OPEN attempts masked per reason (budget/session_budget/soft_gate/hard_gate) + bars with each gate active.
    6. Mechanics hash printed separately from reward hash (reward hash UNCHANGED).
  - Results: **Pending**

# APPEND (2026-02-17): EXPERIMENT-014 COMPLETED + GOAL UPDATE (10%/month)

APPEND (2026-02-17):
- EXPERIMENT-014 completed (Run ID: Final-Train-Test_20260217_124508, reward hash: 09f8bc66, mechanics hash: bea1c04b).
  - Train: avg daily PnL $6.69, PF 1.14, Sharpe 0.78, trades/day 7.3, max intraday DD -233.54.
  - Test: avg daily PnL -$5.72, PF 0.90, Sharpe -0.66, trades/day 6.8, max intraday DD -248.22.
  - Calendar-month DD breaches (Test): Jun peak-DD -$1,509.33; Nov peak-DD -$1,199.92.
  - Session participation: London+NY entries 55.8% achieved, but NY avg PnL/entry negative (-$1.47).

APPEND (2026-02-17): Goals update (authoritative)
- Profit target updated: replace “$50–$100/day” with **~10% per calendar month** on $10k (≈ $1,000/month), evaluated on calendar-month PnL.
- Trade frequency goal relaxed: agent is not required to trade every day; prioritize positive expectancy and DD compliance over activity.
- Risk targets unchanged: daily DD ≤ $500 (in-env), monthly DD ≤ $1,000 (evaluation-only, calendar-month peak DD).
# APPEND (2026-02-17): EXPERIMENT-015 PLANNED

## Quick Comparison Table — EXP-014 row (append)
| EXP-014 | Final-Train-Test_20260217_124508 | -5.72 | 0.90 | -0.66 | 6.8 | N/A | -248.22 | 2,315.04 | Session allocation achieved (London+NY 55.8%); monthly DD breaches Jun (-1509) Nov (-1199); negative test edge. |

APPEND (2026-02-17 — EXP-015 planned):
- EXPERIMENT-015 planned: High-R reward shaping + stop-loss cool-off window + R-distribution diagnostics.
  - Addresses negative Test edge (PF 0.90) and monthly DD breaches (Jun/Nov) from EXP-014.
  - Three minimal changes:
    1. Reward shaping (EXP-015 CHANGE-01): R-normalized high_r_bonus (R≥2.0, cap +0.5) + tiny_win_penalty (0<R<0.2, -0.02). Applied at trade close only.
    2. Mechanics (EXP-015 CHANGE-02): Replace EXP-014 soft gate (1 entry/session after 2 SL) with fixed STOPLOSS_COOLOFF_BARS=45 window after 2nd SL exit. Hard gate at 4 SL unchanged.
    3. Diagnostics (EXP-015 CHANGE-03): R distribution table (7 bins), Session×R summary, gate integrity cross-check printed in TXT log.
  - Success criteria:
    - Calendar-month peak DD ≤ $1,000 all Test months.
    - Target ~10% per calendar month (≈ $1,000/month) while maintaining daily DD compliance.
  - Results: **Pending**