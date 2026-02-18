"""
Phase 2A: Trading Environment Mechanics
Gym-compatible environment for intraday US100 CFD trading simulation.
Handles execution, position sizing, stop-loss enforcement, and constraints.
No reward logic - purely mechanical simulation.

EXPERIMENT-012: Feature-name column mapping + ATR fallback + stop-loss integrity
EXPERIMENT-014: Session-aware entry budget allocation + two-stage stop-loss risk-off gate
EXPERIMENT-015: High-R reward shaping + stop-loss cool-off window (replaces EXP-014 soft gate)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum


# Minimal Gym interface (to avoid external dependencies)
class Space:
    """Base space class."""
    pass


class Discrete(Space):
    """Discrete action space."""
    def __init__(self, n: int):
        self.n = n


class Box(Space):
    """Continuous space."""
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class Env:
    """Base environment class."""
    def __init__(self):
        self.action_space = None
        self.observation_space = None
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return None, {}
    
    def step(self, action):
        return None, 0.0, False, False, {}


class Action(IntEnum):
    """Discrete action space for trading."""
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE_POSITION = 3


class PositionState(IntEnum):
    """Position states."""
    FLAT = 0
    LONG = 1
    SHORT = 2


@dataclass
class Position:
    """Container for active position data."""
    state: PositionState
    entry_price: float
    entry_bar: int
    stop_price: float
    lot_size: float
    risk_amount: float  # Risk amount used for this specific trade (0.5% of equity at entry)
    mae: float = 0.0  # Maximum Adverse Excursion (worst unrealized loss)
    # EXPERIMENT-012: Add provenance fields
    atr_used: float = 0.0  # ATR_14 value at entry (after fallback if applicable)
    atr_was_fallback: bool = False  # Whether ATR fallback was applied
    stop_distance_price: float = 0.0  # Stop distance in price units
    stop_distance_pips: float = 0.0  # Stop distance in pips
    # EXPERIMENT-014: Track which session the entry was made in
    entry_session: str = 'Unknown'
    
    def get_unrealized_pnl(self, current_price: float, pip_size: float = 1.0, pip_value_per_lot: float = 1.0) -> float:
        """
        Calculate unrealized PnL based on current price using pip-based formula.
        
        Args:
            current_price: Current market price
            pip_size: Size of 1 pip in price units (default 1.0 for US100)
            pip_value_per_lot: Dollar value of 1 pip per lot (default 1.0 for US100)
        """
        if self.state == PositionState.LONG:
            price_diff = current_price - self.entry_price
        elif self.state == PositionState.SHORT:
            price_diff = self.entry_price - current_price
        else:
            return 0.0
        
        # Convert price difference to pips and calculate PnL
        price_diff_pips = price_diff / pip_size
        return price_diff_pips * self.lot_size * pip_value_per_lot
    
    def update_mae(self, current_price: float, pip_size: float = 1.0, pip_value_per_lot: float = 1.0) -> None:
        """Update Maximum Adverse Excursion if current unrealized PnL is worse."""
        unrealized_pnl = self.get_unrealized_pnl(current_price, pip_size, pip_value_per_lot)
        if unrealized_pnl < 0:  # Only track losses
            self.mae = max(self.mae, abs(unrealized_pnl))
    
    def get_duration(self, current_bar: int) -> int:
        """Get position duration in bars."""
        return current_bar - self.entry_bar


class TradingEnvironment(Env):
    """
    Gym-compatible trading environment for US100 CFD intraday trading.
    
    Features:
    - Realistic execution (next bar open)
    - Hard stop-loss enforcement (ATR-based)
    - Position sizing based on fixed risk
    - Time constraints (max 90 min per trade, EOD close)
    - Proper lookahead prevention
    
    EXPERIMENT-012: Feature-name column mapping + ATR fallback + stop-loss integrity
    EXPERIMENT-014: Session-aware entry budget allocation + two-stage stop-loss risk-off gate
    EXPERIMENT-015: High-R reward shaping + stop-loss cool-off window (replaces soft gate)
    """
    
    # Environment constants
    INITIAL_BALANCE = 10_000.0
    RISK_PERCENTAGE = 0.005  # 0.5% risk per trade (applied to current equity)
    MAX_TRADE_DURATION = 90  # minutes (bars)
    ATR_PERIOD = 14
    # EXPERIMENT-009: widen stop distance to reduce stop-loss churn
    ATR_MULTIPLIER = 3.0  # Changed from 2.0
    
    # EXPERIMENT-009: manual exit execution convention flag for hash provenance
    MANUAL_EXIT_AT_OPEN = True  # Manual closes execute at next bar OPEN (not CLOSE)
    
    # Broker specifications for US100 CFD
    PIP_SIZE = 1.0           # 1 pip = 1.0 price units
    PIP_VALUE_PER_LOT = 1.0  # $1 per pip per lot
    MIN_LOT_SIZE = 0.01      # Minimum tradeable lot
    MAX_LOT_SIZE = 10.0     # Maximum lot size for risk control (NOT broker max)
    
    # Reward function coefficients
    ALPHA = 0.15  # MAE penalty coefficient
    BETA = 0.2   # Duration penalty coefficient (for losing/breakeven trades)
    GAMMA = 0.05 # Overtrading penalty coefficient
    DAILY_DD_THRESHOLD = 0.05  # 5% daily drawdown threshold
    DAILY_DD_PENALTY = -5.0    # Large penalty for exceeding daily drawdown
    
    # NEW: Reward stability and behavior control parameters
    MIN_HOLD_BARS = 15          # Minimum desired holding time (bars) - UNCHANGED
    HOLD_PENALTY_WEIGHT = 0.30  # Penalty weight for early exits (INCREASED from 0.15)
    ACTION_PENALTY = 0.20       # Fixed penalty for opening/closing trades (INCREASED from 0.05)
    # EXPERIMENT-010: Reduce patience reward proportionally to prevent flat-step reward dominance
    PATIENCE_REWARD = 0.004     # Small reward for staying flat (REDUCED from 0.02; keep once per flat streak behavior)
    DAILY_TRADE_THRESHOLD = 20    # Target maximum trades per day
    OVERTRADE_PENALTY_RATE = 0.01 # Progressive penalty rate per excess trade
    REWARD_CLIP_MIN = -5.0      # Minimum reward value (stability)
    REWARD_CLIP_MAX = 5.0       # Maximum reward value (stability)
    
    # Opportunity-cost-based trade selectivity parameters
    # EXPERIMENT-010: Reduce inactivity baseline substantially to prevent shaping dominance over trade_reward
    INACTIVITY_BASELINE_REWARD = 0.00025  # Baseline reward for staying flat (REDUCED 10x from 0.0025)
    ENTRY_OPPORTUNITY_COST = 0.01         # Opportunity cost penalty for entering trades
    
    # EXPERIMENT-002: Daily trade budget constraint
    MAX_TRADES_PER_DAY = 10  # Hard cap on number of entries per day
    BUDGET_LAMBDA = 0.0      # Convex budget penalty coefficient (EXPERIMENT-003: disabled)
    
    # EXPERIMENT-003: End-of-day unused-budget penalty
    MIN_TRADES_TARGET = 5    # Minimum desired entries/day (soft floor) - EXPERIMENT-004: Increased from 3
    UNUSED_LAMBDA = 0.2      # Unused-budget penalty coefficient
    
    # EXPERIMENT-005: R-normalized execution penalty
    EXECUTION_PENALTY_R = 0.02  # Penalty per execution event, in R units (risk-normalized)
    ACTION_PENALTY_R = 0.01     # Penalty per non-HOLD action, in R units
    
    # EXPERIMENT-007: Stop-loss event penalty (smooth, per-event)
    STOPLOSS_EVENT_PENALTY_R = 0.05  # Penalty per stop-loss exit, in R units
    
    # EXPERIMENT-011: Reversal churn penalty (targeted)
    REVERSAL_EVENT_PENALTY_R = 0.03  # Penalty per reversal exit, in R units (conservative, comparable to execution penalty)
    
    # EXPERIMENT-013: Stop-loss cooldown gate to prevent sustained loss cascades
    STOPLOSS_COOLDOWN_THRESHOLD = 4  # Mask new entries after K stop-loss exits in same day
    
    # EXPERIMENT-014: Session-aware entry budget allocation constants
    RESERVED_ENTRIES_LONDON = 3  # Reserve at least this many entries for London session
    RESERVED_ENTRIES_NY = 4      # Reserve at least this many entries for NY session
    
    # EXPERIMENT-014: Two-stage stop-loss risk-off gate (soft gate replaced by EXP-015 cool-off)
    STOPLOSS_SOFT_GATE_THRESHOLD = 2  # After this many SL exits, apply soft gate (1 entry/session)

    # EXPERIMENT-015: Stop-loss cool-off window (replaces EXP-014 soft gate = 1 entry/session)
    STOPLOSS_COOLOFF_BARS = 45    # Bars to block new entries after 2nd SL exit
    # EXPERIMENT-015: High-R reward shaping coefficients (R-normalized, bounded)
    HIGH_R_BONUS = 0.15           # Bonus coefficient for realized R >= 2.0 (capped at +0.5 total)
    TINY_WIN_PENALTY = 0.02       # Fixed penalty for tiny wins (0 < R < 0.2)
    
    # Column indices for day data (OHLC + features)
    COL_OPEN = 0
    COL_HIGH = 1
    COL_LOW = 2
    COL_CLOSE = 3
    # EXPERIMENT-012: Remove hard-coded COL_ATR; use feature-name mapping instead
    
    def __init__(self, data_loader, window_size: int = 60):
        """
        Initialize trading environment.
        
        Args:
            data_loader: Data loader with get_episode() method and get_feature_index_map()
            window_size: Number of bars in rolling window for state
        """
        super().__init__()
        
        self.data_loader = data_loader
        self.window_size = window_size
        
        # EXPERIMENT-012: Get feature index map from data loader
        if hasattr(data_loader, 'get_feature_index_map'):
            feature_map = data_loader.get_feature_index_map()
            # Map feature names to column indices (after OHLC, so +4)
            self._atr14_col = 4 + feature_map['ATR_14']
            self._atrnorm_col = 4 + feature_map['ATR_norm']
            self._session_asia_col = 4 + feature_map['session_asia']
            self._session_london_col = 4 + feature_map['session_london']
            self._session_ny_col = 4 + feature_map['session_ny']
        else:
            raise ValueError("Data loader must provide get_feature_index_map() method")
        
        # Get episode data
        self.current_day_data = self.data_loader.get_episode()
        self.episode_length = len(self.current_day_data)
        
        # Validate minimum episode length
        if self.episode_length < self.window_size:
            raise ValueError(
                f"Episode length ({self.episode_length}) must be >= window_size ({self.window_size})"
            )
        
        # Environment state
        self.current_bar_idx = self.window_size - 1  # Start after warm-up window
        self.position: Optional[Position] = None
        self.balance = self.INITIAL_BALANCE
        self.realized_pnl = 0.0
        self.peak_balance = self.INITIAL_BALANCE
        
        # Episode tracking
        self.episode_count = 0
        self.trades_history = []
        
        # NEW: Position age tracking (Modification #1)
        self.position_age = 0  # Bars since position was opened
        
        # EXPERIMENT-002: Daily trade budget tracking
        self.trades_taken_today = 0  # Count of entries (OPEN actions that execute)
        
        # EXPERIMENT-007: Remove stop-loss cluster counters (replaced with per-event penalty)
        # (No cluster counters needed anymore)
        
        # EXPERIMENT-012: ATR fallback tracking
        self.atr_fallback_count = 0  # Count of times ATR fallback was applied
        
        # EXPERIMENT-012: Stop-loss integrity tracking
        self.stoploss_positive_pnl_count = 0  # Count of stop_loss exits with positive PnL
        self.exit_price_source_counts = {
            'stop_price': 0,
            'bar_open': 0,
            'bar_close': 0
        }
        
        # EXPERIMENT-012: Action distribution tracking
        self.action_counts = {
            Action.HOLD: 0,
            Action.OPEN_LONG: 0,
            Action.OPEN_SHORT: 0,
            Action.CLOSE_POSITION: 0
        }
        self.masked_open_count = 0  # Opens blocked due to budget
        self.blocked_reversal_count = 0  # Reversals blocked due to MIN_HOLD_BARS
        self.blocked_manual_close_count = 0  # Manual closes blocked due to MIN_HOLD_BARS
        
        # EXPERIMENT-013: Stop-loss cooldown gate tracking
        self.stoploss_exits_today = 0  # Count of stop-loss exits in current episode
        self.cooldown_triggered = False  # Whether cooldown was triggered in current episode
        self.masked_open_cooldown_count = 0  # Opens blocked due to cooldown
        
        # EXPERIMENT-014: Session-aware entry budget allocation counters
        self.entries_in_asia = 0     # Entries made during Asia session
        self.entries_in_london = 0   # Entries made during London session
        self.entries_in_ny = 0       # Entries made during NY session
        
        # EXPERIMENT-014: Session tracking for budget allocation
        # Tracks the "furthest" session reached today: 0=Asia/unknown, 1=London, 2=NY
        self._furthest_session_today = 0
        
        # EXPERIMENT-014: Session budget block tracking
        self.session_budget_block_triggered = False  # Whether session budget block was ever triggered
        self.masked_open_session_budget_count = 0    # OPEN attempts blocked by session budget
        self.masked_open_session_budget_attempts = 0 # Total OPEN attempts while session budget was blocking
        
        # EXPERIMENT-014: Two-stage stop-loss gate tracking
        self.soft_gate_current_session = 'Unknown'   # Session active when soft gate was triggered / last reset
        self.soft_gate_current_session_entries = 0    # Entries in current session since soft gate active
        self.bars_soft_gate_active = 0               # Bars where soft gate (2 SL) was active
        self.bars_hard_gate_active = 0               # Bars where hard gate (4 SL) was active
        self.masked_open_soft_gate_attempts = 0      # OPEN attempts blocked by soft gate
        self.masked_open_hard_gate_attempts = 0      # OPEN attempts blocked by hard gate

        # EXPERIMENT-015: Cool-off window tracking (replaces EXP-014 soft gate)
        self.cooloff_bars_remaining = 0              # Bars remaining in cool-off window
        self.bars_cooloff_active = 0                 # Total bars cool-off was active this episode
        self.masked_open_cooloff_attempts = 0        # OPEN attempts blocked by cool-off window
        
        # Internal flags for deterministic penalty application (Modification #1)
        self.trade_opened_this_step = False
        self.trade_closed_this_step = False
        
        # Daily DD penalty (applied only once per episode)
        self.daily_dd_penalty_applied = False
        
        # Patience reward state
        self.patience_reward_given = False  # Track if patience reward was given during current flat streak
        
        # Action spaces
        self.action_space = Discrete(4)  # 4 actions: HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION
        
        # Observation space (features only)
        num_features = self.current_day_data.shape[1] - 4  # Exclude OHLC columns
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, num_features),
            dtype=np.float32
        )
        
        # Store last action for reward calculation
        self._last_action = None
        
        # Store last reward components for debugging
        self._last_reward_components = {}
    
    # EXPERIMENT-012: ATR fallback helper
    def _get_atr14_for_bar(self, bar_data: np.ndarray) -> Tuple[float, bool]:
        """
        Get ATR_14 value for a bar with fallback rule.
        
        EXPERIMENT-012: If ATR_14 < 1.0 or non-finite, use fallback value of 4.0.
        
        Args:
            bar_data: Single bar of day data (OHLC + features)
        
        Returns:
            Tuple of (atr_value, atr_was_fallback)
        """
        atr = bar_data[self._atr14_col]
        
        # Apply fallback rule
        if not np.isfinite(atr) or atr < 1.0:
            return 4.0, True
        
        return atr, False
    
    # EXPERIMENT-014: Session helpers
    def _get_session_code(self, bar_data: np.ndarray) -> int:
        """
        Get session code from bar data.
        Priority: NY(2) > London(1) > Asia(0).
        
        Args:
            bar_data: Single bar of day data (OHLC + features)
        
        Returns:
            int: 0=Asia/Unknown, 1=London, 2=NY
        """
        ny = bar_data[self._session_ny_col]
        london = bar_data[self._session_london_col]
        
        if ny > 0.5:
            return 2
        elif london > 0.5:
            return 1
        else:
            return 0  # Asia or unknown
    
    def _get_current_session(self) -> str:
        """
        Get current session name for the current bar.
        Priority: NY > London > Asia.
        
        Returns:
            str: 'Asia', 'London', 'NY', or 'Unknown'
        """
        if self.current_bar_idx >= self.episode_length:
            return 'Unknown'
        bar_data = self.current_day_data[self.current_bar_idx]
        ny = bar_data[self._session_ny_col]
        london = bar_data[self._session_london_col]
        asia = bar_data[self._session_asia_col]
        if ny > 0.5:
            return 'NY'
        elif london > 0.5:
            return 'London'
        elif asia > 0.5:
            return 'Asia'
        else:
            return 'Unknown'
    
    # EXPERIMENT-014: Session budget allocation check
    def _is_session_budget_blocked(self) -> bool:
        """
        Check if OPEN is blocked by session-aware budget allocation.
        
        Asia:   trades_taken_today >= MAX - RESERVED_LONDON - RESERVED_NY (while London/NY not yet reached)
        London: trades_taken_today >= MAX - RESERVED_NY (while NY not yet reached)
        NY:     normal budget cap only (no session-specific block)
        
        Returns:
            bool: True if OPEN should be blocked by session budget allocation
        """
        # EXPERIMENT-014: Check if we're still in Asia (no London or NY reached yet)
        if self._furthest_session_today < 1:
            # Asia: reserve entries for both London and NY
            asia_limit = self.MAX_TRADES_PER_DAY - self.RESERVED_ENTRIES_LONDON - self.RESERVED_ENTRIES_NY
            return self.trades_taken_today >= asia_limit
        elif self._furthest_session_today < 2:
            # London (but NY not yet reached): reserve entries for NY
            london_limit = self.MAX_TRADES_PER_DAY - self.RESERVED_ENTRIES_NY
            return self.trades_taken_today >= london_limit
        # In NY: no session budget block; normal MAX_TRADES_PER_DAY cap applies
        return False
    
    # EXPERIMENT-014: Soft gate check
    def _is_soft_gate_blocked(self) -> bool:
        """
        Check if OPEN is blocked by the soft stop-loss gate.
        
        After 2 stop-loss exits (but before hard gate at 4), allow at most 1 entry
        per session until session changes.
        
        Returns:
            bool: True if OPEN should be blocked by soft gate
        """
        # EXPERIMENT-014: Soft gate active between threshold 2 and hard gate 4
        if self.STOPLOSS_SOFT_GATE_THRESHOLD <= self.stoploss_exits_today < self.STOPLOSS_COOLDOWN_THRESHOLD:
            return self.soft_gate_current_session_entries >= 1
        return False
    
    # EXPERIMENT-014: Unified open-allowed check
    def _check_open_allowed(self) -> Tuple[bool, str]:
        """
        Check if an OPEN action is allowed, with reason if blocked.
        Priority: budget > hard_gate > session_budget > cooloff
        EXPERIMENT-015: Replaced soft_gate with cooloff window check.

        Returns:
            Tuple of (allowed: bool, reason_if_blocked: str or None)
        """
        if self.trades_taken_today >= self.MAX_TRADES_PER_DAY:
            return False, 'budget'
        if self.stoploss_exits_today >= self.STOPLOSS_COOLDOWN_THRESHOLD:
            return False, 'hard_gate'
        if self._is_session_budget_blocked():
            return False, 'session_budget'
        # EXPERIMENT-015: Replace soft gate (1 entry/session) with fixed cool-off window
        if self.cooloff_bars_remaining > 0:
            return False, 'cooloff'
        return True, None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed (optional)
        
        Returns:
            Tuple of (initial_state, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode data
        self.current_day_data = self.data_loader.get_episode()
        self.episode_length = len(self.current_day_data)
        
        # Validate minimum episode length
        if self.episode_length < self.window_size:
            raise ValueError(
                f"Episode length ({self.episode_length}) must be >= window_size ({self.window_size})"
            )
        
        # Reset environment state
        self.current_bar_idx = self.window_size - 1
        self.position = None
        self.balance = self.INITIAL_BALANCE
        self.realized_pnl = 0.0
        self.peak_balance = self.INITIAL_BALANCE
        
        # Reset episode tracking
        self.episode_count += 1
        self.trades_history = []
        
        # Reset position age
        self.position_age = 0
        
        # EXPERIMENT-002: Reset daily budget tracking
        self.trades_taken_today = 0
        
        # EXPERIMENT-007: No cluster counters to reset
        
        # EXPERIMENT-012: Reset ATR fallback tracking
        self.atr_fallback_count = 0
        
        # EXPERIMENT-012: Reset stop-loss integrity tracking
        self.stoploss_positive_pnl_count = 0
        self.exit_price_source_counts = {
            'stop_price': 0,
            'bar_open': 0,
            'bar_close': 0
        }
        
        # EXPERIMENT-012: Reset action tracking
        self.action_counts = {
            Action.HOLD: 0,
            Action.OPEN_LONG: 0,
            Action.OPEN_SHORT: 0,
            Action.CLOSE_POSITION: 0
        }
        self.masked_open_count = 0
        self.blocked_reversal_count = 0
        self.blocked_manual_close_count = 0
        
        # EXPERIMENT-013: Reset cooldown tracking
        self.stoploss_exits_today = 0
        self.cooldown_triggered = False
        self.masked_open_cooldown_count = 0
        
        # EXPERIMENT-014: Reset session entry counters
        self.entries_in_asia = 0
        self.entries_in_london = 0
        self.entries_in_ny = 0
        
        # EXPERIMENT-014: Reset session tracking state
        self._furthest_session_today = 0
        
        # EXPERIMENT-014: Reset session budget block tracking
        self.session_budget_block_triggered = False
        self.masked_open_session_budget_count = 0
        self.masked_open_session_budget_attempts = 0
        
        # EXPERIMENT-014: Reset soft/hard gate tracking
        self.soft_gate_current_session = 'Unknown'
        self.soft_gate_current_session_entries = 0
        self.bars_soft_gate_active = 0
        self.bars_hard_gate_active = 0
        self.masked_open_soft_gate_attempts = 0
        self.masked_open_hard_gate_attempts = 0

        # EXPERIMENT-015: Reset cool-off window tracking
        self.cooloff_bars_remaining = 0
        self.bars_cooloff_active = 0
        self.masked_open_cooloff_attempts = 0
        
        # Reset flags
        self.trade_opened_this_step = False
        self.trade_closed_this_step = False
        self.daily_dd_penalty_applied = False
        self.patience_reward_given = False
        
        # Reset action tracking
        self._last_action = None
        self._last_reward_components = {}
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in environment.
        
        Args:
            action: Action to take (0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3=CLOSE)
        
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Store action for reward calculation
        self._last_action = action
        
        # EXPERIMENT-012: Track action distribution
        action_enum = Action(action)
        self.action_counts[action_enum] += 1
        
        # Reset execution flags at start of each step
        self.trade_opened_this_step = False
        self.trade_closed_this_step = False
        
        # Track end-of-day
        self.is_eod = (self.current_bar_idx >= self.episode_length - 1)
        
        # Advance time FIRST (next bar)
        self.current_bar_idx += 1
        
        # Update position age if position is active
        if self.position is not None:
            self.position_age += 1
        
        # Check if we've exhausted the episode
        if self.current_bar_idx >= self.episode_length:
            # Episode ends - return terminal state
            state = self._get_state()
            reward = self._calculate_reward()
            info = self._get_info()
            return state, reward, True, False, info
        
        # Get current bar (we're now at t+1 after advancing)
        current_bar = self.current_day_data[self.current_bar_idx]
        
        # EXPERIMENT-014: Update session tracking for current bar
        session_code = self._get_session_code(current_bar)
        self._furthest_session_today = max(self._furthest_session_today, session_code)
        
        # EXPERIMENT-014: Update soft gate session tracking — reset per-session entries on session change
        soft_gate_active = (self.STOPLOSS_SOFT_GATE_THRESHOLD <= self.stoploss_exits_today < self.STOPLOSS_COOLDOWN_THRESHOLD)
        if soft_gate_active:
            current_session_str = self._get_current_session()
            if current_session_str != self.soft_gate_current_session:
                # Session changed — reset per-session entry allowance
                self.soft_gate_current_session = current_session_str
                self.soft_gate_current_session_entries = 0
            self.bars_soft_gate_active += 1
        
        # EXPERIMENT-015: Decrement cool-off window; track active bars
        if self.cooloff_bars_remaining > 0:
            self.cooloff_bars_remaining -= 1
            self.bars_cooloff_active += 1
        
        # EXPERIMENT-014: Track hard gate bars
        if self.stoploss_exits_today >= self.STOPLOSS_COOLDOWN_THRESHOLD:
            self.bars_hard_gate_active += 1
        
        # Update MAE if position exists
        if self.position is not None:
            current_price = current_bar[self.COL_CLOSE]
            self.position.update_mae(current_price, self.PIP_SIZE, self.PIP_VALUE_PER_LOT)
        
        # Check stop-loss BEFORE agent action
        stop_hit = self._check_stop_loss()
        
        # If not stopped out, execute agent action
        if not stop_hit:
            
            if action_enum == Action.HOLD:
                pass  # Do nothing
            
            elif action_enum == Action.OPEN_LONG:
                # EXPERIMENT-014: Use unified check (budget + cooldown + session_budget + soft_gate)
                open_allowed, block_reason = self._check_open_allowed()
                
                if open_allowed:
                    if self.position is None:
                        # Open new long position
                        self._open_position(PositionState.LONG)
                    else:
                        # EXPERIMENT-011: enforce MIN_HOLD_BARS on reversals
                        trade_duration = self.position.get_duration(self.current_bar_idx)
                        if trade_duration >= self.MIN_HOLD_BARS:
                            # Reverse: close current, open long
                            self._close_position(reason="reversal")
                            self._open_position(PositionState.LONG)
                        else:
                            # EXPERIMENT-012: Track blocked reversal
                            self.blocked_reversal_count += 1
                else:
                    # EXPERIMENT-014: Attribute block to the correct reason
                    if block_reason == 'budget':
                        self.masked_open_count += 1
                    elif block_reason == 'hard_gate':
                        self.masked_open_cooldown_count += 1
                        self.masked_open_hard_gate_attempts += 1
                        # EXPERIMENT-013: Mark cooldown triggered
                        if not self.cooldown_triggered:
                            self.cooldown_triggered = True
                    elif block_reason == 'session_budget':
                        # EXPERIMENT-014: Session budget block
                        self.masked_open_session_budget_count += 1
                        self.masked_open_session_budget_attempts += 1
                        self.session_budget_block_triggered = True
                    elif block_reason == 'soft_gate':
                        # EXPERIMENT-014: Soft gate block
                        self.masked_open_soft_gate_attempts += 1
                    elif block_reason == 'cooloff':
                        # EXPERIMENT-015: Cool-off window block
                        self.masked_open_cooloff_attempts += 1
            
            elif action_enum == Action.OPEN_SHORT:
                # EXPERIMENT-014: Use unified check (budget + cooldown + session_budget + soft_gate)
                open_allowed, block_reason = self._check_open_allowed()
                
                if open_allowed:
                    if self.position is None:
                        # Open new short position
                        self._open_position(PositionState.SHORT)
                    else:
                        # EXPERIMENT-011: enforce MIN_HOLD_BARS on reversals
                        trade_duration = self.position.get_duration(self.current_bar_idx)
                        if trade_duration >= self.MIN_HOLD_BARS:
                            # Reverse: close current, open short
                            self._close_position(reason="reversal")
                            self._open_position(PositionState.SHORT)
                        else:
                            # EXPERIMENT-012: Track blocked reversal
                            self.blocked_reversal_count += 1
                else:
                    # EXPERIMENT-014: Attribute block to the correct reason
                    if block_reason == 'budget':
                        self.masked_open_count += 1
                    elif block_reason == 'hard_gate':
                        self.masked_open_cooldown_count += 1
                        self.masked_open_hard_gate_attempts += 1
                        if not self.cooldown_triggered:
                            self.cooldown_triggered = True
                    elif block_reason == 'session_budget':
                        # EXPERIMENT-014: Session budget block
                        self.masked_open_session_budget_count += 1
                        self.masked_open_session_budget_attempts += 1
                        self.session_budget_block_triggered = True
                    elif block_reason == 'soft_gate':
                        # EXPERIMENT-014: Soft gate block
                        self.masked_open_soft_gate_attempts += 1
                    elif block_reason == 'cooloff':
                        # EXPERIMENT-015: Cool-off window block
                        self.masked_open_cooloff_attempts += 1
            
            elif action_enum == Action.CLOSE_POSITION:
                if self.position is not None:
                    trade_duration = self.position.get_duration(self.current_bar_idx)
                    
                    # NEW (Modification #1): Enforce minimum hold period with hard override
                    if trade_duration >= self.MIN_HOLD_BARS:
                        # Allow close only if minimum hold satisfied
                        self._close_position(reason="manual")
                    else:
                        # EXPERIMENT-012: Track blocked manual close
                        self.blocked_manual_close_count += 1
        
        # Check time-based constraints
        self._check_time_constraints()
        
        # Force close at end of day
        if self.current_bar_idx >= self.episode_length - 1:
            if self.position is not None:
                self._force_close_position(reason="end_of_day")
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        state = self._get_state()
        
        # Check termination
        terminated = (self.current_bar_idx >= self.episode_length - 1)
        truncated = False
        
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask (which actions are valid).
        
        EXPERIMENT-002: Mask OPEN actions when daily budget is exhausted.
        EXPERIMENT-013: Mask OPEN actions when stop-loss cooldown is active.
        EXPERIMENT-014: Mask OPEN actions when session budget allocation blocks entry.
        EXPERIMENT-014: Mask OPEN actions when soft stop-loss gate is active.
        
        Returns:
            Boolean array of shape (4,) indicating valid actions
        """
        mask = np.ones(4, dtype=bool)
        
        # EXPERIMENT-002: Mask OPEN_LONG and OPEN_SHORT if budget exhausted
        if self.trades_taken_today >= self.MAX_TRADES_PER_DAY:
            mask[Action.OPEN_LONG] = False
            mask[Action.OPEN_SHORT] = False
        
        # EXPERIMENT-013: Mask OPEN_LONG and OPEN_SHORT if hard cooldown is active
        if self.stoploss_exits_today >= self.STOPLOSS_COOLDOWN_THRESHOLD:
            mask[Action.OPEN_LONG] = False
            mask[Action.OPEN_SHORT] = False
            if not self.cooldown_triggered:
                self.cooldown_triggered = True  # Mark cooldown as triggered for this episode
        
        # EXPERIMENT-014: Mask OPEN if session budget allocation blocks entry
        if self._is_session_budget_blocked():
            mask[Action.OPEN_LONG] = False
            mask[Action.OPEN_SHORT] = False
        
        # EXPERIMENT-015: Mask OPEN if cool-off window is active (replaces EXP-014 soft gate)
        if self.cooloff_bars_remaining > 0:
            mask[Action.OPEN_LONG] = False
            mask[Action.OPEN_SHORT] = False
        
        # Can't close if no position
        if self.position is None:
            mask[Action.CLOSE_POSITION] = False
        
        return mask
    
    def _open_position(self, position_type: PositionState):
        """
        Open a new position.
        
        Args:
            position_type: LONG or SHORT
        """
        # Entry price: current bar open (we've already advanced to next bar)
        current_bar = self.current_day_data[self.current_bar_idx]
        entry_price = current_bar[self.COL_OPEN]
        
        # EXPERIMENT-012: Calculate stop-loss using ATR with fallback rule
        atr, atr_was_fallback = self._get_atr14_for_bar(current_bar)
        if atr_was_fallback:
            self.atr_fallback_count += 1
        
        stop_distance = self.ATR_MULTIPLIER * atr
        
        if position_type == PositionState.LONG:
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
        
        # Calculate position size using CURRENT EQUITY (dynamic risk)
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            stop_distance = entry_price * 0.01
        
        # Convert stop distance from price units to pips
        stop_distance_pips = stop_distance / self.PIP_SIZE
        
        # Calculate risk amount (0.5% of current equity) and lot size
        risk_amount = self.balance * self.RISK_PERCENTAGE
        lot_size = risk_amount / (stop_distance_pips * self.PIP_VALUE_PER_LOT)
        
        # Apply broker constraints
        lot_size = np.clip(lot_size, self.MIN_LOT_SIZE, self.MAX_LOT_SIZE)
        
        # EXPERIMENT-014: Capture entry session
        entry_session = self._get_current_session()
        
        # EXPERIMENT-012: Create position with provenance fields
        # EXPERIMENT-014: Store entry_session in position
        self.position = Position(
            state=position_type,
            entry_price=entry_price,
            entry_bar=self.current_bar_idx,
            stop_price=stop_price,
            lot_size=lot_size,
            risk_amount=risk_amount,  # Store for trade-specific reward normalization
            atr_used=atr,  # EXPERIMENT-012
            atr_was_fallback=atr_was_fallback,  # EXPERIMENT-012
            stop_distance_price=stop_distance,  # EXPERIMENT-012
            stop_distance_pips=stop_distance_pips,  # EXPERIMENT-012
            entry_session=entry_session  # EXPERIMENT-014
        )
        
        # EXPERIMENT-014: Track session-specific entry counts
        if entry_session == 'Asia':
            self.entries_in_asia += 1
        elif entry_session == 'London':
            self.entries_in_london += 1
        elif entry_session == 'NY':
            self.entries_in_ny += 1
        
        # EXPERIMENT-014: If soft gate is active, count this entry toward per-session limit
        if self.STOPLOSS_SOFT_GATE_THRESHOLD <= self.stoploss_exits_today < self.STOPLOSS_COOLDOWN_THRESHOLD:
            # Initialize soft gate session tracking if not yet set
            if self.soft_gate_current_session == 'Unknown':
                self.soft_gate_current_session = entry_session
            self.soft_gate_current_session_entries += 1
        
        # NEW: Reset position age when opening a new position
        self.position_age = 0
        
        # EXPERIMENT-002: Increment trade count when entry actually executes
        self.trades_taken_today += 1
        
        # FIXED (Modification #1 refinement): Set flag to signal trade was opened
        self.trade_opened_this_step = True
    
    def _close_position(self, exit_price: Optional[float] = None, reason: str = "manual"):
        """
        Close current position and record trade.
        
        Args:
            exit_price: Specific exit price (for stops), or None for current bar
            reason: Reason for close (manual, stop, time, eod, reversal)
        """
        if self.position is None:
            return
        
        # EXPERIMENT-012: Determine exit_price_source for integrity tracking
        exit_price_source = "bar_close"  # Default
        
        # Exit price: current bar close/open (we've already advanced) or specified for stops
        if exit_price is None:
            current_bar = self.current_day_data[self.current_bar_idx]
            # EXPERIMENT-009: manual exit executes at next bar OPEN for consistency with entry execution
            if reason == "manual":
                exit_price = current_bar[self.COL_OPEN]
                exit_price_source = "bar_open"  # EXPERIMENT-012
            else:
                # For other forced closes (max_duration, end_of_day), use CLOSE
                exit_price = current_bar[self.COL_CLOSE]
                exit_price_source = "bar_close"  # EXPERIMENT-012
        else:
            # Explicit exit_price provided (stop-loss case)
            exit_price_source = "stop_price"  # EXPERIMENT-012
        
        # EXPERIMENT-012: Track exit_price_source usage
        self.exit_price_source_counts[exit_price_source] += 1
        
        # Calculate price difference in price units
        if self.position.state == PositionState.LONG:
            price_diff = exit_price - self.position.entry_price
        else:  # SHORT
            price_diff = self.position.entry_price - exit_price
        
        # Convert price difference to pips
        price_diff_pips = price_diff / self.PIP_SIZE
        
        # Calculate realized PnL using proper pip-based formula
        pnl = price_diff_pips * self.position.lot_size * self.PIP_VALUE_PER_LOT
        
        # CRITICAL: Validate PnL to prevent NaN/inf propagation
        if not np.isfinite(pnl):
            pnl = 0.0
        
        # EXPERIMENT-012: Stop-loss integrity check
        if reason == "stop_loss" and pnl > 0.01:
            self.stoploss_positive_pnl_count += 1
            print(f"[STOP-LOSS INTEGRITY WARNING] Date: {getattr(self.data_loader, 'date', 'unknown')}, "
                  f"Bar: {self.current_bar_idx}, Entry: {self.position.entry_price:.2f}, "
                  f"Stop: {self.position.stop_price:.2f}, Exit: {exit_price:.2f}, "
                  f"PnL: ${pnl:.2f}, Direction: {self.position.state.name}, "
                  f"exit_price_source: {exit_price_source}")
        
        # Update accounting
        self.realized_pnl += pnl
        self.balance += pnl
        
        # CRITICAL: Validate balance to prevent NaN/inf propagation
        if not np.isfinite(self.balance):
            self.balance = self.INITIAL_BALANCE
        if not np.isfinite(self.realized_pnl):
            self.realized_pnl = 0.0
        
        # Update peak balance for daily drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # EXPERIMENT-012: Record trade with provenance fields
        # EXPERIMENT-014: Include entry_session in trade_record
        trade_record = {
            'entry_bar': self.position.entry_bar,
            'exit_bar': self.current_bar_idx,
            'position_type': self.position.state.name,
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'stop_price': self.position.stop_price,
            'lot_size': self.position.lot_size,
            'pnl': pnl,
            'mae': self.position.mae,
            'duration': self.position.get_duration(self.current_bar_idx),
            'exit_reason': reason,
            'risk_amount': self.position.risk_amount,
            # EXPERIMENT-012: Provenance fields
            'exit_price_source': exit_price_source,
            'atr_used': self.position.atr_used,
            'atr_was_fallback': self.position.atr_was_fallback,
            'stop_distance_price': self.position.stop_distance_price,
            'stop_distance_pips': self.position.stop_distance_pips,
            # EXPERIMENT-014: Session attribution
            'entry_session': self.position.entry_session,
        }
        self.trades_history.append(trade_record)
        
        # EXPERIMENT-013: Track stop-loss exits for cooldown gate
        if reason == "stop_loss":
            self.stoploss_exits_today += 1
            # EXPERIMENT-015: Set cool-off window when the 2nd SL exit occurs
            if self.stoploss_exits_today == 2:
                self.cooloff_bars_remaining = self.STOPLOSS_COOLOFF_BARS
            # EXPERIMENT-014: When soft gate threshold is first crossed, initialize session tracking
            if self.stoploss_exits_today == self.STOPLOSS_SOFT_GATE_THRESHOLD:
                # Soft gate just became active: initialize to current session so entries reset
                self.soft_gate_current_session = self._get_current_session()
                self.soft_gate_current_session_entries = 0
        
        # Clear position
        self.position = None
        
        # NEW: Reset position age when position is closed
        self.position_age = 0
        
        # FIXED (Modification #1 refinement): Set flag to signal trade was closed
        self.trade_closed_this_step = True
    
    def _check_stop_loss(self) -> bool:
        """
        Check if stop-loss was hit and enforce if necessary.
        
        Returns:
            bool: True if stop was hit
        """
        if self.position is None:
            return False
        
        current_bar = self.current_day_data[self.current_bar_idx]
        low = current_bar[self.COL_LOW]
        high = current_bar[self.COL_HIGH]
        
        # Check long stop
        if self.position.state == PositionState.LONG:
            if low <= self.position.stop_price:
                self._close_position(
                    exit_price=self.position.stop_price,
                    reason="stop_loss"
                )
                return True
        
        # Check short stop
        elif self.position.state == PositionState.SHORT:
            if high >= self.position.stop_price:
                self._close_position(
                    exit_price=self.position.stop_price,
                    reason="stop_loss"
                )
                return True
        
        return False
    
    def _check_time_constraints(self) -> bool:
        """
        Check if position has exceeded max duration and force close if needed.
        
        Returns:
            bool: True if time-based exit occurred
        """
        if self.position is None:
            return False
        
        duration = self.position.get_duration(self.current_bar_idx)
        
        if duration >= self.MAX_TRADE_DURATION:
            self._close_position(reason="max_duration")
            return True
        
        return False
    
    def _calculate_progressive_overtrade_penalty(self) -> float:
        """
        Calculate progressive penalty for excessive daily trading.
        
        Returns:
            float: Penalty value (<= 0)
        """
        if self.trades_taken_today <= self.DAILY_TRADE_THRESHOLD:
            return 0.0
        
        excess_trades = self.trades_taken_today - self.DAILY_TRADE_THRESHOLD
        penalty = -self.OVERTRADE_PENALTY_RATE * (excess_trades ** 1.5)
        
        return penalty
    
    def _force_close_position(self, reason: str):
        """Force close position (for EOD, etc)."""
        if self.position is not None:
            self._close_position(reason=reason)
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state observation from rolling window.
        
        Returns:
            np.ndarray of shape (window_size, 29) - features only, not OHLC
        """
        start_idx = self.current_bar_idx - self.window_size + 1
        end_idx = self.current_bar_idx
        
        state_window = self.current_day_data[start_idx:end_idx + 1]
        features_only = state_window[:, 4:]  # Skip first 4 columns (OHLC)
        
        return features_only.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary for current step."""
        info = {
            'episode': self.episode_count,
            'bar': self.current_bar_idx,
            'balance': self.balance,
            'realized_pnl': self.realized_pnl,
            'peak_balance': self.peak_balance,
            'position': self.position.state.name if self.position else 'FLAT',
            'position_age': self.position_age,
            'trades_today': self.trades_taken_today,
            'trade_opened_this_step': self.trade_opened_this_step,
            'trade_closed_this_step': self.trade_closed_this_step,
            'reward_components': self._last_reward_components  # For debugging
        }
        
        return info
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for completed episode."""
        if not self.trades_history:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_duration': 0.0,
                'exit_reasons': {}
            }
        
        total_trades = len(self.trades_history)
        winning_trades = sum(1 for t in self.trades_history if t['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(t['pnl'] for t in self.trades_history)
        avg_duration = np.mean([t['duration'] for t in self.trades_history])
        
        # Count exit reasons
        exit_reasons = {}
        for trade in self.trades_history:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_duration': avg_duration,
            'exit_reasons': exit_reasons
        }
    
    def _calculate_reward(self) -> float:
        """
        Calculate step reward with R-normalized components.
        
        Design principles:
        - All penalties/rewards expressed in R-multiples (risk units)
        - Reward function is account-size agnostic
        - Enables fair comparison across different equity levels
        
        Returns:
            float: Clipped reward value
        """
        components = {}
        total_reward = 0.0
        
        # Component 1: Trade reward (R-normalized)
        if self.trade_closed_this_step:
            last_trade = self.trades_history[-1]
            pnl = last_trade['pnl']
            risk_amount = last_trade['risk_amount']
            
            if risk_amount > 0:
                trade_reward = pnl / risk_amount
            else:
                trade_reward = 0.0
            
            components['trade_reward'] = trade_reward
            total_reward += trade_reward
            
            # Component 2: MAE penalty (capped at 1.0 R)
            if last_trade['duration'] >= 2:
                mae = last_trade['mae']
                mae_penalty = -self.ALPHA * min(mae / risk_amount, 1.0) if risk_amount > 0 else 0.0
                components['mae_penalty'] = mae_penalty
                total_reward += mae_penalty
            else:
                components['mae_penalty'] = 0.0
            
            # Component 3: Duration penalty (for losing/breakeven trades only)
            if pnl <= 0:
                duration = last_trade['duration']
                duration_penalty = -self.BETA * (duration / self.MAX_TRADE_DURATION)
                components['duration_penalty'] = duration_penalty
                total_reward += duration_penalty
            else:
                components['duration_penalty'] = 0.0
            
            # Component 4: Early exit penalty (EXPERIMENT-010: manual closes only)
            if last_trade['exit_reason'] == "manual" and pnl < 0:
                duration = last_trade['duration']
                if duration < self.MIN_HOLD_BARS:
                    early_exit_penalty = -self.HOLD_PENALTY_WEIGHT * (self.MIN_HOLD_BARS - duration) / self.MIN_HOLD_BARS
                    components['early_exit_penalty'] = early_exit_penalty
                    total_reward += early_exit_penalty
                else:
                    components['early_exit_penalty'] = 0.0
            else:
                components['early_exit_penalty'] = 0.0
            
            # Component 5: Entry opportunity cost
            entry_cost = -self.ENTRY_OPPORTUNITY_COST
            components['entry_cost'] = entry_cost
            total_reward += entry_cost
            
            # Component 6: Execution penalty (R-normalized)
            execution_penalty = -2 * self.EXECUTION_PENALTY_R  # Two events: open + close
            components['execution_penalty'] = execution_penalty
            total_reward += execution_penalty
            
            # Component 7: Stop-loss event penalty (EXPERIMENT-007)
            if last_trade['exit_reason'] == "stop_loss":
                stoploss_penalty = -self.STOPLOSS_EVENT_PENALTY_R
                components['stoploss_event_penalty'] = stoploss_penalty
                total_reward += stoploss_penalty
            else:
                components['stoploss_event_penalty'] = 0.0
            
            # Component 8: Reversal event penalty (EXPERIMENT-011)
            if last_trade['exit_reason'] == "reversal":
                reversal_penalty = -self.REVERSAL_EVENT_PENALTY_R
                components['reversal_event_penalty'] = reversal_penalty
                total_reward += reversal_penalty
            else:
                components['reversal_event_penalty'] = 0.0
            
            # EXPERIMENT-015: High-R bonus (R-normalized, bounded)
            # Reward large-R outcomes to encourage high R:R trades
            if risk_amount > 0:
                R = pnl / risk_amount  # Same as trade_reward
            else:
                R = 0.0
            if R >= 2.0:
                high_r_bonus = min(self.HIGH_R_BONUS * (R - 2.0), 0.5)  # Hard cap at +0.5
                components['high_r_bonus'] = high_r_bonus
                total_reward += high_r_bonus
            else:
                components['high_r_bonus'] = 0.0
            
            # EXPERIMENT-015: Tiny-win penalty (R-normalized, fixed)
            # Suppress insignificant wins to discourage noise trades
            if 0.0 < R < 0.2:
                tiny_win_penalty = -self.TINY_WIN_PENALTY
                components['tiny_win_penalty'] = tiny_win_penalty
                total_reward += tiny_win_penalty
            else:
                components['tiny_win_penalty'] = 0.0
        else:
            # No trade closed
            components['trade_reward'] = 0.0
            components['mae_penalty'] = 0.0
            components['duration_penalty'] = 0.0
            components['early_exit_penalty'] = 0.0
            components['entry_cost'] = 0.0
            components['execution_penalty'] = 0.0
            components['stoploss_event_penalty'] = 0.0
            components['reversal_event_penalty'] = 0.0
            components['high_r_bonus'] = 0.0       # EXPERIMENT-015
            components['tiny_win_penalty'] = 0.0   # EXPERIMENT-015
        
        # Component 9: Action penalty (R-normalized, scaled by exposure)
        if self._last_action != Action.HOLD and self._last_action is not None:
            if self.position is not None:
                exposure_scale = self.position.lot_size / self.MAX_LOT_SIZE
            else:
                exposure_scale = 0.1
            
            action_penalty = -self.ACTION_PENALTY_R * exposure_scale
            components['action_penalty'] = action_penalty
            total_reward += action_penalty
        else:
            components['action_penalty'] = 0.0
        
        # Component 10: Patience reward (EXPERIMENT-010)
        if self._last_action == Action.HOLD and self.position is None:
            if not self.patience_reward_given:
                patience_reward = self.PATIENCE_REWARD
                components['patience_reward'] = patience_reward
                total_reward += patience_reward
                self.patience_reward_given = True
            else:
                components['patience_reward'] = 0.0
        else:
            self.patience_reward_given = False
            components['patience_reward'] = 0.0
        
        # Component 11: Inactivity baseline (EXPERIMENT-010)
        if self._last_action == Action.HOLD and self.position is None:
            inactivity_reward = self.INACTIVITY_BASELINE_REWARD
            components['inactivity_baseline'] = inactivity_reward
            total_reward += inactivity_reward
        else:
            components['inactivity_baseline'] = 0.0
        
        # Component 12: Progressive overtrading penalty
        overtrading_penalty = self._calculate_progressive_overtrade_penalty()
        components['overtrading_penalty'] = overtrading_penalty
        total_reward += overtrading_penalty
        
        # Component 13: Daily DD penalty (applied once per episode if triggered)
        current_dd = self.INITIAL_BALANCE - self.balance
        current_dd_pct = current_dd / self.INITIAL_BALANCE
        
        if current_dd_pct > self.DAILY_DD_THRESHOLD and not self.daily_dd_penalty_applied:
            daily_dd_penalty = self.DAILY_DD_PENALTY
            components['daily_dd_penalty'] = daily_dd_penalty
            total_reward += daily_dd_penalty
            self.daily_dd_penalty_applied = True
        else:
            components['daily_dd_penalty'] = 0.0
        
        # Component 14: End-of-day unused budget penalty (EXPERIMENT-003)
        if self.is_eod and self.trades_taken_today < self.MIN_TRADES_TARGET:
            unused_penalty = -self.UNUSED_LAMBDA * (self.MIN_TRADES_TARGET - self.trades_taken_today)
            components['unused_budget_penalty'] = unused_penalty
            total_reward += unused_penalty
        else:
            components['unused_budget_penalty'] = 0.0
        
        # Store components for debugging
        self._last_reward_components = components
        
        # Clip reward for stability
        clipped_reward = np.clip(total_reward, self.REWARD_CLIP_MIN, self.REWARD_CLIP_MAX)
        
        return float(clipped_reward)