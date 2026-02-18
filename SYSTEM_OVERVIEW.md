# SYSTEM OVERVIEW (MODEL-FACING)

Last updated: 2026-02-16

## 1) Goal
FTMO-style prop-firm compliance on $10k account:
- Trades/day: 3–10 desired (hard cap at 10)
- Daily DD: ≤ 5% of $10k start-of-day ($500) – in-env penalty
- Monthly DD: ≤ 10% ($1,000) – evaluation-only tracking (must be calendar-month correct)
- Profit target: $50–$100/day avg

## 2) Data pipeline
- Source: US100_RL_features.parquet (Timestamp + OHLC + 29 features)
- Trading-hours filter: Mon–Thu 01:05–23:50, Fri 01:05–22:55, Sat/Sun excluded
- Episode = one calendar day after filtering; must have ≥ 60 bars
- State to agent = rolling 60-bar window of features only: shape (60, 29)
- Env internal day array = OHLC + features: shape (n_bars, 32)

IMPORTANT (EXP-012 implemented; correctness baseline):
- Do not rely on hard-coded feature indices (e.g., ATR_14). Use feature-name-to-index mapping to prevent column drift bugs.
- Mandated ATR fallback rule: if ATR_14 < 1.0 (or non-finite), use ATR_14 = 4.0 for stop-loss and ATR-dependent computations.
- Stop-loss integrity: track and report any stop_loss exits with positive PnL (should be zero or near-zero with explanations).
- All diagnostics use feature-name mapping for session flags and volatility metrics.

## 3) Environment (TradingEnvironment in trading_env.py)
**Actions (4):** HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION  
**Execution:** decide at bar t → advance to t+1 → execute at next bar open/close

**Risk + constraints:**
- Stop distance = ATR_MULTIPLIER × ATR_14 (EXP-009: 3.0)
- Risk per trade = 0.5% of current equity at entry
- Lot size clipped [0.01, 10.0]
- Min hold enforced: 15 bars (hard override – close before 15 bars → HOLD; enforced on reversals since EXP-011)
- Max duration: 90 bars → forced close
- EOD forced close
- Manual exit execution (EXP-009): Manual CLOSE executes at next bar OPEN

**Trade budget:**
- MAX_TRADES_PER_DAY = 10 entries/day (entries only, closes always allowed)
- OPEN actions masked when budget exhausted
- Stop-loss cooldown gate (EXP-013): OPEN actions masked for remainder of day after ≥4 stop-loss exits (STOPLOSS_COOLDOWN_THRESHOLD = 4)
- EOD unused-budget penalty if trades < MIN_TRADES_TARGET (5)

**Reward components (R-normalized where applicable):**
1. Trade reward: realized_pnl / trade_risk (R-multiples)
2. MAE penalty: -0.15 × min(MAE/trade_risk, 1.0) [if duration ≥ 2]
3. Duration penalty: -0.2 × (bars_held / 90) [losing trades only]
4. Progressive overtrading penalty (as implemented)
5. Early exit penalty: -0.30 × (15 - duration) [losing trades, manual exits only]
6. Stop-loss event penalty: -0.05 R [each stop-loss exit]
7. Reversal event penalty: -0.03 R [each reversal exit] (EXP-011)
8. Execution penalty: -0.02 R per open/close event
9. Action penalty: -0.01 R × exposure_scale per non-HOLD action
10. Patience reward: +0.004 for first HOLD while FLAT
11. Inactivity baseline: +0.00025 per HOLD while FLAT
12. Entry opportunity cost: -0.01 per entry
13. Daily DD penalty: -5.0 once if DD from INITIAL_BALANCE > 5%
14. Unused budget penalty: -0.2 �� (MIN_TRADES_TARGET - trades) at EOD
- Final reward clipped to [-5, +5]

## 4) Agent (DQNAgent in dqn_agent.py)
- QR-DQN (Quantile Regression DQN) with risk-averse action selection
- Outputs 51 quantile estimates per action
- Risk-averse action selection: argmax over mean of bottom 25% quantiles
- Quantile regression loss with Huber threshold κ=1.0
- Double DQN, Dueling architecture, 1D CNN encoder
- Replay buffer: 250k uniform, batch 64
- LR: 1e-4 (Adam), gamma: 0.99
- Target network hard update every 5000 update_steps
- CQL: disabled
- Epsilon: 3-phase schedule using env_steps (reaches 0.01 by ~85% of training)
- Eval: deterministic argmax

## 5) Training pipeline (train.py + run_final.py)
- Phase A (burn-in): epsilon calibration via observed_steps_per_day
- Phase B (walk-forward): expanding folds
- Phase C (final): Train 2021–2024, Test 2025, exports all test trades to CSV
- TXT logging includes: calendar-month DD table (Test), top-10 worst days table (Test), reward component aggregates (Test)
- TXT logging includes: trades/day histogram, trade duration by exit_reason, trade-level win rate
- TXT logging includes: stop-loss intensity histogram, volatility quintile table, session slice summary
- TXT logging includes: action distribution + blocked actions, ATR fallback usage, stop-loss integrity, exit-price-source distribution (EXP-012)

## 6) Known code issues
- EXP-012 correctness issues resolved (ATR mapping, stop-loss labeling). Current issues are primarily performance/regime-related.
- Session slice may still be degenerate or timezone-misaligned; require session-flag sanity diagnostics before session-based tuning.

# APPEND (2026-02-17): Goal update (profit target framing; authoritative)
- Profit target updated: **~10% per calendar month** on $10k (≈ $1,000/month), evaluated on calendar-month PnL.
- Trades/day is not a strict target; agent may trade 0–10 entries/day; focus on positive expectancy and DD compliance.
- Risk targets unchanged: daily DD ≤ $500 (in-env), monthly DD ≤ $1,000 (evaluation-only; calendar-month peak DD).
# APPEND (2026-02-17): EXPERIMENT-015 reward components and mechanics

## New reward component keys (EXP-015; append to Section 3 reward components list):
- high_r_bonus: +HIGH_R_BONUS × max(R − 2.0, 0), capped at +0.5; applied at trade close when R ≥ 2.0. Coefficient HIGH_R_BONUS = 0.15. R-normalized, bounded.
- tiny_win_penalty: −TINY_WIN_PENALTY (= −0.02) fixed; applied at trade close when 0 < R < 0.2. R-normalized.

## Mechanics update (EXP-015):
- Stop-loss cool-off window: after 2nd SL exit per day, masks OPEN_LONG/OPEN_SHORT for STOPLOSS_COOLOFF_BARS = 45 bars. Replaces EXP-014 soft gate (1 entry/session). Hard gate at 4 SL exits/day unchanged.
- New tracking: cooloff_bars_remaining, bars_cooloff_active, masked_open_cooloff_attempts.

## Updated goal (supersedes Section 1 profit target; authoritative):
- Profit target: **~10% per calendar month** on $10k (≈ $1,000/month), evaluated on calendar-month PnL.
- Trade frequency: not a strict target; agent may trade 0–10 entries/day; prioritize positive expectancy over activity.
- Risk limits unchanged: daily DD ≤ $500 (in-env), monthly DD ≤ $1,000 (evaluation-only, calendar-month peak DD).