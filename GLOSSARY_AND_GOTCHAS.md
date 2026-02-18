# GLOSSARY + GOTCHAS (MODEL-FACING)

## Glossary
| Term | Definition |
|---|---|
| Episode | One filtered trading day |
| State | (60, 29) features-only rolling window |
| Internal day array | (n_bars, 32) = OHLC(4) + features(29) |
| R / R-multiple | Trade PnL ÷ risk_amount; 1R = one unit of risk (0.5% of equity at entry) |
| Trade budget | Counts entries only (OPEN_LONG/OPEN_SHORT executions); closes do not consume budget |
| Cap-hit day | A day where trades_taken_today reached MAX_TRADES_PER_DAY |
| MAE | Maximum Adverse Excursion – worst unrealized loss during a trade |
| Exposure scale | lot_size / MAX_LOT_SIZE; used to scale action penalty |
| env_steps | Environment interaction steps (every action selected during training) – EXPERIMENT-006 |
| update_steps | Optimizer update steps (every train_step() call) – EXPERIMENT-006 |
| Stop-loss event penalty | Per-event penalty applied when stop-loss occurs – EXPERIMENT-007 |
| QR-DQN / Quantiles | Quantile Regression DQN – learns quantile estimates of return distribution (Dabney et al. 2018) – EXPERIMENT-008 |
| Risk-averse action selection | Select actions by mean of lowest k quantiles (bottom 25%) – EXPERIMENT-008 |
| Risk statistic | Mean of bottom k quantiles used for argmax action selection in QR-DQN – EXPERIMENT-008 |
| Calendar-month DD | Peak-to-trough drawdown computed within each calendar month; required for FTMO monthly limit checks (evaluation-only) |
| Manual exit at next bar OPEN | Execution convention where agent's manual CLOSE (and reversal close leg) executes at the next bar OPEN, matching entry execution timing. |
| Reward dominance | Shaping terms contribute more to total reward than trade_reward, risking misalignment with profitability objective. |
| Reversal churn loophole | Reversals bypass min-hold intent via OPEN_* while in a position; fixed in EXP-011. |
| Reversal event penalty | Per-event R-normalized penalty applied when exit_reason == "reversal" – EXPERIMENT-011 |
| Exit reason integrity check | Diagnostic invariant verifying that certain exit reasons (e.g., stop_loss) match expected PnL sign/price-source semantics; targeted in EXP-012 |
| Feature-name column mapping | Mapping from feature column names (e.g., ATR_14, session_ny) to indices in the env day array; required to avoid hard-coded index drift bugs; implemented in EXP-012 |
| ATR fallback rule | If ATR_14 < 1.0 (or non-finite), use ATR_14 = 4.0 for stop-loss and ATR-dependent computations; mandated in EXP-012 |
| Exit price source | Provenance field tracking which price was used for exit: stop_price, bar_open, or bar_close; added in EXP-012 |
| ATR fallback count | Counter tracking how many times ATR fallback rule was applied; added in EXP-012 |
| Stoploss positive PnL count | Counter tracking stop_loss exits with positive PnL (integrity violation); added in EXP-012 |
| Stop-loss cooldown gate | (Implemented EXP-013) Mechanics constraint that masks new OPEN actions for the remainder of the day after ≥K stop_loss exits (K=4), to prevent sustained loss cascades; not a reward term. |
| Session-aware entry budget allocation | (Implemented EXP-014) Mechanics constraint that reserves a minimum number of daily entries for later sessions (London/NY) so Asia cannot consume the full daily cap; implemented via OPEN action masking; not a reward term. Constants: RESERVED_ENTRIES_LONDON=3, RESERVED_ENTRIES_NY=4. Asia limit = MAX-3-4=3; London limit = MAX-4=6; NY = full cap. |
| Cap timing | (Implemented EXP-014 diagnostics) The session / bar-index (or timestamp) at which the Nth daily entry occurs (especially the 10th entry), used to verify whether the daily cap is consumed before London/NY. |
| Two-stage stop-loss risk-off gate | (Implemented EXP-014) Soft gate at 2 SL exits/day: allows at most 1 additional entry per session until session changes; hard gate at 4 SL exits/day (EXP-013 cooldown). Both are action masking only, no reward terms. |
| Soft gate | (EXP-014) Stop-loss risk-off gate activated after STOPLOSS_SOFT_GATE_THRESHOLD (=2) SL exits/day; allows 1 entry per session (resets on session change) until hard gate threshold is reached. |
| Mechanics hash | (EXP-014) A separate MD5 hash of action-masking/allocation constants (MAX_TRADES_PER_DAY, STOPLOSS_COOLDOWN_THRESHOLD, RESERVED_ENTRIES_LONDON, RESERVED_ENTRIES_NY, STOPLOSS_SOFT_GATE_THRESHOLD) printed in TXT logs without changing the existing reward_function_hash semantics. |
| entry_session | (EXP-014) Field stored in each trade_record indicating which session (Asia/London/NY/Unknown) the entry was made in; set at _open_position() time using live session flags; used for accurate entries-by-session aggregation in diagnostics without re-inference. |

## Must-not-break interfaces
- `TradingEnvironment.reset()` calls `data_loader.get_episode()`
- `RLDataLoader` does NOT implement `get_episode()`
- `EpisodeDataLoader` (in train.py) wraps RLDataLoader and provides `get_episode()` → required when instantiating `TradingEnvironment`

## Known code drift / issues
Precedence: CURRENT_STATE.md > executed code > logs > other docs.

| Issue | Location | Status |
|---|---|---|
| Session slice summary appears degenerate (Asia dominates, NY missing) | train.py session slice diagnostics | FIXED (EXP-014): Old session slice summary marked deprecated in TXT log. New Entries-by-session table uses trade_record[entry_session] (set at open execution) for accurate attribution. |
| Cooldown "masked opens (cooldown)" counter stays 0 even when cooldown triggers | trading_env.py + train.py diagnostics | FIXED (EXP-014): Unified _check_open_allowed() helper correctly attributes each blocked OPEN to budget/hard_gate/session_budget/soft_gate. masked_open_hard_gate_attempts and masked_open_soft_gate_attempts added. |

## Reward scaling requirement (project constraint)
All reward/penalty terms MUST be:
- Risk-normalized (R units) and invariant to account size and risk%
- Compatible with reward clipping [-5, +5]
- Any new term must follow this convention

## Experiment change protocol
1. Change as few things as possible per experiment
2. Document hypothesis before running
3. Compare against previous experiment's results
4. Update CURRENT_STATE.md and EXPERIMENTS_LOG.md after each run
5. Never edit completed experiment entries (append-only)

## Open Issues (EXP-014)
| Issue | Location | Status |
|---|---|---|
| Session distribution is not yet measured post-EXP-014 | TXT log from next run | OPEN: EXP-014 adds the diagnostics to measure it. If London+NY combined < 40%, consider increasing RESERVED_ENTRIES_LONDON/NY. |
| Whether soft gate improves Sep 2025 monthly DD | Test 2025 results | OPEN: pending execution of EXP-014. Hypothesis is that 2 SL exits → 1-entry-per-session soft gate + session allocation will reduce stop-loss cluster cascades in peak-loss days. |
| Train session diagnostics require full re-evaluation pass | train.py | OPEN: Train env objects not retained post-training. EXP-014 diagnostics print Train session budget constants only; a separate eval pass on Train dates would be needed for full Train session attribution. |

# APPEND (2026-02-17): Goal definition update (profit target framing)
- Profit target is now tracked as **~10% per calendar month** on $10k (≈ $1,000/month), evaluated on calendar-month PnL.
- Trade count is not a target; agent may trade 0–10 entries/day; focus on positive expectancy and DD compliance.

# APPEND (2026-02-17): Proposed new diagnostic term (for EXP-015)
- "R distribution table": aggregated counts of realized trade R-multiples in bins (e.g., >=2R, >=3R, <=-1R). This is required to validate any “high-R bonus” shaping without reward hacking.
# APPEND (2026-02-17): EXP-015 new glossary entries

| Term | Definition |
|---|---|
| high_r_bonus | (EXP-015 reward shaping) R-normalized bonus applied at trade close when realized R ≥ 2.0. Formula: min(HIGH_R_BONUS × (R − 2.0), 0.5). Coefficient HIGH_R_BONUS = 0.15; hard cap on bonus contribution = +0.5. Included in components dict as key `high_r_bonus`. Expressed in R-units; compatible with reward clip [−5, +5]. |
| tiny_win_penalty | (EXP-015 reward shaping) Fixed R-normalized penalty (−TINY_WIN_PENALTY = −0.02) applied at trade close when 0 < R < 0.2. Suppresses insignificant wins to discourage noise trades. Included in components dict as key `tiny_win_penalty`. Applied only on trade close. |
| stop-loss cool-off window | (EXP-015 mechanics) Replaces EXP-014 soft gate. After the 2nd stop-loss exit in a day (stoploss_exits_today == 2), sets cooloff_bars_remaining = STOPLOSS_COOLOFF_BARS (= 45). While cooloff_bars_remaining > 0, OPEN_LONG and OPEN_SHORT are masked. Decremented by 1 each step. Hard gate at 4 SL exits (EXP-013) unchanged. Tracked via bars_cooloff_active and masked_open_cooloff_attempts. |
| R distribution table | (EXP-015 diagnostic) Binned counts of realized trade R-multiples (<= -1R, (-1R, -0.2R], (-0.2R, 0.2R), [0.2R, 1R), [1R, 2R), >= 2R, >= 3R) printed in TXT log. Required to validate high-R bonus shaping and detect reward hacking (e.g., agent chasing tiny gains in >=2R bin at expense of overall edge). |
| cooloff_bars_remaining | (EXP-015) Integer counter tracking remaining bars in the stop-loss cool-off window. Set to STOPLOSS_COOLOFF_BARS on the 2nd SL exit of the day; decremented by 1 each step; cannot go below 0. While > 0, OPEN actions are blocked. |
| bars_cooloff_active | (EXP-015) Episode counter tracking how many bars the cool-off window was active. Incremented each step where cooloff_bars_remaining > 0 before decrement. Used in gate integrity diagnostic. |
| masked_open_cooloff_attempts | (EXP-015) Episode counter tracking OPEN_LONG or OPEN_SHORT attempts that were blocked specifically by the cool-off window (block_reason == 'cooloff'). Used in gate integrity diagnostic. |