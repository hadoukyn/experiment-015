"""
Train script for DQN trading agent with feature curriculum and walk-forward validation.
Corrected version with proper API integration and comprehensive metadata tracking.
"""

import os
import sys
import json
import warnings
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import csv
from pathlib import Path

# Import project modules - CORRECTED: Use actual class name
from data_loader import RLDataLoader as DataLoader
from trading_env import TradingEnvironment
from dqn_agent import DQNAgent

warnings.filterwarnings('ignore')


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class DualLogger:
    """Logger that writes to both console and file simultaneously."""
    
    def __init__(self, log_file: str):
        """
        Initialize dual logger.
        
        Args:
            log_file: Path to log file
        """
        self.terminal = sys.stdout
        # Open with UTF-8 encoding to handle Unicode characters on Windows
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """Close the log file."""
        self.log.close()


def setup_phase_logging(phase_name: str, log_dir: str = "logs") -> DualLogger:
    """
    Setup logging for a training phase.
    
    Args:
        phase_name: Name of the phase (e.g., "Burn-in", "Walk-Forward", "Final-Train")
        log_dir: Directory to store log files
    
    Returns:
        DualLogger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{phase_name}_{timestamp}.txt")
    
    logger = DualLogger(log_file)
    sys.stdout = logger
    
    print(f"\n{'='*80}")
    print(f"LOGGING INITIALIZED: {phase_name}")
    print(f"Log file: {log_file}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*80}\n")
    
    return logger


def restore_logging(logger: DualLogger):
    """
    Restore standard output and close logger.
    
    Args:
        logger: DualLogger instance to close
    """
    sys.stdout = logger.terminal
    logger.close()


# ============================================================================
# EPSILON DECAY HELPER
# ============================================================================

def calculate_epsilon_decay_steps(
    num_training_days: int,
    observed_steps_per_day: float,
    coverage_ratio: float = 0.85
) -> int:
    """
    Calculate epsilon decay steps based on training length and observed interaction rates.
    
    Ensures epsilon reaches minimum (epsilon_end) at a fixed percentage of training,
    allowing adequate exploration throughout the learning process.
    
    Args:
        num_training_days: Number of days in training set
        observed_steps_per_day: Measured average steps per episode from actual training
        coverage_ratio: Fraction of training where epsilon decays (default 0.85)
            - 0.85 means epsilon reaches minimum at 85% of training
            - Last 15% is pure exploitation for convergence
    
    Returns:
        Number of steps over which to decay epsilon from start to end
    
    Examples:
        - 517 days @ 1342 steps/day: 517 × 1342 × 0.85 = 589,931 steps
        - 710 days @ 1342 steps/day: 710 × 1342 × 0.85 = 809,597 steps
    
    Academic Basis:
        Conservative epsilon schedules prevent premature convergence (Mnih et al. 2015)
        and improve robustness in non-stationary environments (Lussange et al. 2019).
    """
    total_expected_steps = num_training_days * observed_steps_per_day
    decay_steps = int(total_expected_steps * coverage_ratio)
    
    print(f"Epsilon Decay Calculation:")
    print(f"  Training days: {num_training_days}")
    print(f"  Observed steps/day: {observed_steps_per_day:.1f}")
    print(f"  Expected total steps: {total_expected_steps:,.0f}")
    print(f"  Coverage ratio: {coverage_ratio}")
    print(f"  Decay steps: {decay_steps:,} (epsilon reaches min at {coverage_ratio*100:.0f}% of training)")
    
    return decay_steps


# ============================================================================
# FEATURE CURRICULUM DEFINITION
# ============================================================================

FEATURE_PHASES = {
    1: [
        'log_return',
        'momentum_10_norm',
        'roc_5',
        'rolling_std_20',
        'z_score',
        'candle_body_ratio',
        'candle_range_ratio',
        'close_position_ratio',
        'delta_range_norm'
    ],
    2: [
        'ATR_14',
        'ATR_norm',
        'return_quantile_15',
        'dist_sma_20',
        'dist_ema_9',
        'dist_ema_21',
        'regime_vol_score'
    ],
    3: [
        'time_sin',
        'time_cos',
        'session_asia',
        'session_london',
        'session_ny',
        'session_progress_sin',
        'session_progress_cos'
    ],
    4: [
        'dist_sma_100',
        'macd_norm',
        'macd_signal_norm',
        'dist_to_hh_60',
        'dist_to_ll_60'
    ]
}

ALL_FEATURES = [
    'ATR_14',
    'ATR_norm',
    'dist_sma_20',
    'dist_sma_100',
    'dist_ema_9',
    'dist_ema_21',
    'momentum_10_norm',
    'roc_5',
    'rsi_scaled',
    'macd_norm',
    'macd_signal_norm',
    'log_return',
    'rolling_std_20',
    'z_score',
    'candle_body_ratio',
    'candle_range_ratio',
    'close_position_ratio',
    'delta_range_norm',
    'return_quantile_15',
    'time_sin',
    'time_cos',
    'session_asia',
    'session_london',
    'session_ny',
    'session_progress_sin',
    'session_progress_cos',
    'dist_to_hh_60',
    'dist_to_ll_60',
    'regime_vol_score'
]


def get_active_features(max_phase: int) -> List[str]:
    """Get list of features active up to specified phase."""
    active = []
    for phase in range(1, max_phase + 1):
        active.extend(FEATURE_PHASES[phase])
    return active


def create_feature_mask(max_phase: int) -> np.ndarray:
    """Create binary mask for feature selection."""
    active_features = get_active_features(max_phase)
    mask = np.array([1 if f in active_features else 0 for f in ALL_FEATURES], dtype=np.float32)
    return mask


def apply_feature_mask(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply feature mask to state tensor."""
    return state * mask


def get_reward_function_hash() -> str:
    """
    Generate hash of reward function parameters for reproducibility tracking.
    
    EXPERIMENT-008: Now includes EXECUTION_PENALTY_R, ACTION_PENALTY_R, STOPLOSS_EVENT_PENALTY_R
    EXPERIMENT-009: include stop/exit mechanics in hash for reproducibility
    EXPERIMENT-010: expanded to include INACTIVITY_BASELINE_REWARD and PATIENCE_REWARD (changed in EXP-010)
    EXPERIMENT-011: include REVERSAL_EVENT_PENALTY_R for churn penalty tracking
    
    Returns:
        8-character hex hash of reward parameters
    """
    reward_params = (
        f"{TradingEnvironment.ALPHA}_"
        f"{TradingEnvironment.BETA}_"
        f"{TradingEnvironment.GAMMA}_"
        f"{TradingEnvironment.DAILY_DD_THRESHOLD}_"
        f"{TradingEnvironment.DAILY_DD_PENALTY}_"
        f"{TradingEnvironment.RISK_PERCENTAGE}_"
        f"{TradingEnvironment.MAX_TRADE_DURATION}_"
        f"{TradingEnvironment.EXECUTION_PENALTY_R}_"
        f"{TradingEnvironment.ACTION_PENALTY_R}_"
        f"{TradingEnvironment.STOPLOSS_EVENT_PENALTY_R}_"
        f"{TradingEnvironment.ATR_MULTIPLIER}_"  # EXPERIMENT-009
        f"{TradingEnvironment.MANUAL_EXIT_AT_OPEN}_"  # EXPERIMENT-009
        f"{TradingEnvironment.INACTIVITY_BASELINE_REWARD}_"  # EXPERIMENT-010
        f"{TradingEnvironment.PATIENCE_REWARD}_"  # EXPERIMENT-010
        f"{TradingEnvironment.REVERSAL_EVENT_PENALTY_R}"  # EXPERIMENT-011
    )
    return hashlib.md5(reward_params.encode()).hexdigest()[:8]


# EXPERIMENT-014: Mechanics hash (separate from reward hash; covers masking/allocation constants)
def get_mechanics_hash() -> str:
    """
    Generate hash of mechanics constants that affect agent behavior but NOT reward function.
    
    EXPERIMENT-014: Tracks session-aware budget allocation + soft gate constants.
    This is separate from the reward_function_hash to preserve backward compatibility.
    
    Returns:
        8-character hex hash of mechanics parameters
    """
    mechanics_params = (
        f"{TradingEnvironment.MAX_TRADES_PER_DAY}_"
        f"{TradingEnvironment.STOPLOSS_COOLDOWN_THRESHOLD}_"
        f"{TradingEnvironment.RESERVED_ENTRIES_LONDON}_"
        f"{TradingEnvironment.RESERVED_ENTRIES_NY}_"
        f"{TradingEnvironment.STOPLOSS_SOFT_GATE_THRESHOLD}"
    )
    return hashlib.md5(mechanics_params.encode()).hexdigest()[:8]


# ============================================================================
# DATA LOADER WRAPPER FOR ENVIRONMENT COMPATIBILITY
# ============================================================================

class EpisodeDataLoader:
    """
    Wrapper around RLDataLoader to provide get_episode() interface
    compatible with TradingEnvironment.
    
    EXPERIMENT-012: Exposes feature-name-to-index mapping to environment.
    """
    
    def __init__(self, data_loader: DataLoader, date: str):
        """
        Initialize episode data loader.
        
        Args:
            data_loader: RLDataLoader instance
            date: Date string in 'YYYY-MM-DD' format
        """
        self.data_loader = data_loader
        self.date = date
        self._episode_data = None
        
        # EXPERIMENT-012: Store feature columns and create index map
        self.feature_cols = data_loader.feature_cols
        self.feature_index_map = {name: idx for idx, name in enumerate(self.feature_cols)}
        
        self._load_episode()
    
    def _load_episode(self):
        """Load and cache episode data for the specified date."""
        df = self.data_loader.data
        day_df = df[df['Date'].astype(str) == self.date]
        
        if len(day_df) == 0:
            raise ValueError(f"No data found for date {self.date}")
        
        # CRITICAL FIX: Include OHLC + Features
        # We select by column NAME, so Timestamp is automatically excluded
        # Order in DataFrame: [Timestamp, Open, High, Low, Close, ...features...]
        # We want: [Open, High, Low, Close, ...features...]
        
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        ohlc_data = day_df[ohlc_cols].values
        features_data = day_df[self.data_loader.feature_cols].values
        
        # Concatenate: OHLC first, then features
        # Result shape: (n_bars, 32) = 4 OHLC + 29 features
        self._episode_data = np.concatenate([ohlc_data, features_data], axis=1)
    
    def get_episode(self) -> np.ndarray:
        """
        Get episode data (OHLC + features, but NOT Timestamp).
        
        Returns:
            np.ndarray of shape (n_bars, 32) - [Open, High, Low, Close, ...29 features...]
        """
        return self._episode_data
    
    def get_feature_index_map(self) -> Dict[str, int]:
        """
        Get feature-name-to-index mapping.
        
        EXPERIMENT-012: Required by TradingEnvironment to resolve column indices by name.
        
        Returns:
            Dict mapping feature names to indices (0-based, relative to feature array)
        """
        return self.feature_index_map


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_episode(
    env: TradingEnvironment,
    agent: DQNAgent,
    feature_mask: np.ndarray,
    date: str,  # EXPERIMENT-012: Add date parameter
    train_freq: int = 4,
    verbose: bool = False
) -> Dict:
    """Run one training episode."""
    state, info = env.reset()
    state = apply_feature_mask(state, feature_mask)
    
    episode_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0
    losses = []
    td_losses = []
    cql_losses = []
    
    # EXPERIMENT-013: Accumulate reward components
    reward_components_sum = {}
    
    while not (terminated or truncated):
        current_action_mask = env.get_action_mask()
        action = agent.select_action(state, current_action_mask, training=True)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = apply_feature_mask(next_state, feature_mask)
        next_action_mask = env.get_action_mask()

        # Store transition with both current and next action masks
        agent.store_transition(state, action, reward, next_state, 
                             terminated or truncated, 
                             current_action_mask, next_action_mask)
        
        # EXPERIMENT-013: Accumulate reward components if available
        if 'reward_components' in info:
            for key, value in info['reward_components'].items():
                if key != 'trade_just_closed':  # Skip boolean flags
                    reward_components_sum[key] = reward_components_sum.get(key, 0.0) + value
        
        if step_count % train_freq == 0:
            loss_dict = agent.train_step()
            if loss_dict is not None:
                losses.append(loss_dict['total_loss'])
                td_losses.append(loss_dict['td_loss'])
                cql_losses.append(loss_dict['cql_loss'])
        
        state = next_state
        episode_reward += reward
        step_count += 1
    
    # Get episode summary from environment
    summary = env.get_episode_summary()
    
    # Add additional metrics
    metrics = {
        'date': date,  # EXPERIMENT-012: Include date
        'episode_reward': episode_reward,
        'realized_pnl': summary['total_pnl'],
        'max_drawdown': env.balance - env.peak_balance,  # Intraday drawdown
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'avg_mae': np.mean([t['mae'] for t in env.trades_history]) if env.trades_history else 0.0,
        'avg_trade_duration': summary['avg_duration'] if summary['total_trades'] > 0 else 0.0,
        'exit_reasons': summary.get('exit_reasons', {}),
        'steps': step_count,
        'avg_loss': np.mean(losses) if losses else 0.0,
        'avg_td_loss': np.mean(td_losses) if td_losses else 0.0,
        'avg_cql_loss': np.mean(cql_losses) if cql_losses else 0.0,
        'reward_components_sum': reward_components_sum  # EXPERIMENT-013
    }
    
    return metrics


def evaluate_episode(
    env: TradingEnvironment,
    agent: DQNAgent,
    feature_mask: np.ndarray,
    date: str,  # EXPERIMENT-012: Add date parameter
    eval_mode: str = 'argmax',
    temperature: float = 0.15
) -> Dict:
    """
    Run one evaluation episode (no training, epsilon=0).
    
    Args:
        env: Trading environment
        agent: DQN agent
        feature_mask: Feature mask to apply
        date: Date string for this episode
        eval_mode: Evaluation policy mode ('argmax' or 'softmax')
        temperature: Temperature for softmax sampling (only used if eval_mode='softmax')
        
    Returns:
        Dictionary with episode metrics
    """
    state, info = env.reset()
    state = apply_feature_mask(state, feature_mask)
    
    episode_reward = 0.0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action_mask = env.get_action_mask()
        action = agent.select_action(state, action_mask, training=False, 
                                     eval_mode=eval_mode, temperature=temperature)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = apply_feature_mask(next_state, feature_mask)
        
        
        state = next_state
        episode_reward += reward
    
    # Get episode summary from environment
    summary = env.get_episode_summary()
    
    metrics = {
        'date': date,  # EXPERIMENT-012: Include date
        'episode_reward': episode_reward,
        'realized_pnl': summary['total_pnl'],
        'max_drawdown': env.balance - env.peak_balance,
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'avg_mae': np.mean([t['mae'] for t in env.trades_history]) if env.trades_history else 0.0,
        'avg_trade_duration': summary['avg_duration'] if summary['total_trades'] > 0 else 0.0,
        'exit_reasons': summary.get('exit_reasons', {})
    }
    
    return metrics


def print_episode_summary(date: str, metrics: Dict, prefix: str = ""):
    """Print formatted episode summary."""
    print(f"{prefix}Date: {date}")
    print(f"{prefix}  Episode Reward: {metrics['episode_reward']:.2f}")
    print(f"{prefix}  Realized PnL: ${metrics['realized_pnl']:.2f}")
    print(f"{prefix}  Max Drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"{prefix}  Trades: {metrics['total_trades']}")
    print(f"{prefix}  Win Rate: {metrics['win_rate']:.2%}")
    print(f"{prefix}  Avg MAE: ${metrics['avg_mae']:.2f}")
    print(f"{prefix}  Avg Trade Duration: {metrics['avg_trade_duration']:.1f} steps")
    
    exit_reasons = metrics.get('exit_reasons', {})
    if exit_reasons:
        print(f"{prefix}  Exit Reasons: {exit_reasons}")


def calculate_cumulative_equity_drawdown(all_metrics: List[Dict], starting_balance: float = 10_000.0) -> Dict:
    """
    Calculate cumulative equity curve drawdown across episodes.
    
    This tracks the peak-to-trough drawdown of a running equity curve,
    treating each episode's realized PnL as a daily change in capital.
    
    IMPROVED: Now considers intraday drawdowns when reconstructing peak equity.
    
    EVALUATION ONLY - does not affect training or environment.
    
    Args:
        all_metrics: List of episode metrics containing 'realized_pnl' and 'max_drawdown'
        starting_balance: Initial capital (default: $10,000)
    
    Returns:
        Dict containing:
            - running_balance: Final equity after all episodes
            - peak_balance: Highest equity reached
            - max_drawdown: Largest peak-to-trough decline
            - max_drawdown_pct: Largest decline as percentage of peak
            - equity_curve: List of daily equity values
    """
    running_balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0.0
    equity_curve = [starting_balance]
    
    for metrics in all_metrics:
        episode_pnl = metrics['realized_pnl']
        episode_intraday_dd = metrics['max_drawdown']  # Negative value
        
        # Key insight: If episode ended with positive PnL but had intraday losses,
        # the intraday peak was higher than the final balance.
        # We need to consider this peak for accurate cumulative DD calculation.
        
        # Case 1: Episode had intraday drawdown
        if episode_intraday_dd < 0:
            # The intraday peak was at: final_balance - intraday_dd
            # Since intraday_dd is negative, this gives us the peak
            episode_end_balance = running_balance + episode_pnl
            
            # Reconstruct the intraday peak
            # intraday_dd = balance_at_end_of_episode - peak_during_episode
            # Therefore: peak_during_episode = balance_at_end_of_episode - intraday_dd
            intraday_peak = episode_end_balance - episode_intraday_dd
            
            # Update global peak if this intraday peak was higher
            if intraday_peak > peak_balance:
                peak_balance = intraday_peak
            
            # Now update running balance to end-of-episode value
            running_balance = episode_end_balance
            
            # Calculate drawdown from peak to current balance
            current_drawdown = peak_balance - running_balance
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # Case 2: No intraday drawdown (episode_intraday_dd >= 0)
        else:
            # Update running balance
            running_balance += episode_pnl
            
            # Update peak if new high
            if running_balance > peak_balance:
                peak_balance = running_balance
            
            # Calculate current drawdown
            current_drawdown = peak_balance - running_balance
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        equity_curve.append(running_balance)
    
    max_drawdown_pct = (max_drawdown / peak_balance * 100) if peak_balance > 0 else 0.0
    
    return {
        'running_balance': running_balance,
        'peak_balance': peak_balance,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'equity_curve': equity_curve
    }
def calculate_daily_equity_curve(all_metrics: List[Dict]) -> Dict:
    """
    Calculate equity curve based on daily P&L with proper drawdown tracking.
    This does NOT simulate compounding - it tracks actual daily performance.
    
    Args:
        all_metrics: List of episode metrics with 'realized_pnl' key
        
    Returns:
        Dictionary with equity curve statistics
    """
    if not all_metrics:
        return {
            'daily_pnls': [],
            'daily_balances': [],
            'cumulative_pnl': 0.0,
            'max_daily_win': 0.0,
            'max_daily_loss': 0.0,
            'winning_days': 0,
            'losing_days': 0,
            'win_rate_daily': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'peak_cumulative_pnl': 0.0,
            'max_drawdown_from_peak': 0.0,
            'current_drawdown': 0.0
        }
    
    daily_pnls = [m['realized_pnl'] for m in all_metrics]
    
    # Calculate cumulative equity
    cumulative_pnl = 0.0
    daily_balances = []
    peak_pnl = 0.0
    max_drawdown = 0.0
    
    for pnl in daily_pnls:
        cumulative_pnl += pnl
        daily_balances.append(cumulative_pnl)
        
        # Update peak
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        
        # Calculate drawdown from peak
        current_dd = peak_pnl - cumulative_pnl
        if current_dd > max_drawdown:
            max_drawdown = current_dd
    
    # Win/Loss statistics
    wins = [pnl for pnl in daily_pnls if pnl > 0]
    losses = [pnl for pnl in daily_pnls if pnl < 0]
    
    winning_days = len(wins)
    losing_days = len(losses)
    total_days = len(daily_pnls)
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Consecutive streaks
    max_consec_wins = 0
    max_consec_losses = 0
    current_streak = 0
    last_was_win = None
    
    for pnl in daily_pnls:
        is_win = pnl > 0
        if last_was_win is None or last_was_win == is_win:
            current_streak += 1
        else:
            current_streak = 1
        
        if is_win:
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            max_consec_losses = max(max_consec_losses, current_streak)
        
        last_was_win = is_win
    
    return {
        'daily_pnls': daily_pnls,
        'daily_balances': daily_balances,
        'cumulative_pnl': cumulative_pnl,
        'max_daily_win': max(daily_pnls),
        'max_daily_loss': min(daily_pnls),
        'winning_days': winning_days,
        'losing_days': losing_days,
        'win_rate_daily': winning_days / total_days if total_days > 0 else 0.0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consec_wins,
        'max_consecutive_losses': max_consec_losses,
        'peak_cumulative_pnl': peak_pnl,
        'max_drawdown_from_peak': max_drawdown,
        'current_drawdown': peak_pnl - cumulative_pnl
    }


def print_daily_performance_summary(phase_name: str, all_metrics: List[Dict]) -> Dict:
    """
    Print performance summary based on daily P&L tracking.
    This replaces print_phase_summary for better clarity.
    
    Returns:
        Dictionary with equity statistics
    """
    if not all_metrics:
        return {}
    
    # Calculate daily equity curve
    equity_stats = calculate_daily_equity_curve(all_metrics)
    
    # Original metrics
    total_pnl = sum(m['realized_pnl'] for m in all_metrics)
    max_intraday_dd = min(m['max_drawdown'] for m in all_metrics)
    total_trades = sum(m['total_trades'] for m in all_metrics)
    num_days = len(all_metrics)
    
    print(f"\n{'='*80}")
    print(f"{phase_name} PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n--- Trading Activity ---")
    print(f"Trading Days: {num_days}")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades/Day: {total_trades / num_days:.1f}")
    
    print(f"\n--- Daily P&L Performance ---")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average Daily P&L: ${total_pnl / num_days:,.2f}")
    print(f"Best Day: ${equity_stats['max_daily_win']:,.2f}")
    print(f"Worst Day: ${equity_stats['max_daily_loss']:,.2f}")
    print(f"Daily P&L Std Dev: ${np.std(equity_stats['daily_pnls']):,.2f}")
    
    print(f"\n--- Win/Loss Statistics ---")
    print(f"Winning Days: {equity_stats['winning_days']} ({equity_stats['win_rate_daily']:.1%})")
    print(f"Losing Days: {equity_stats['losing_days']}")
    print(f"Average Win: ${equity_stats['avg_win']:,.2f}")
    print(f"Average Loss: ${equity_stats['avg_loss']:,.2f}")
    print(f"Profit Factor: {equity_stats['profit_factor']:.2f}")
    print(f"Max Consecutive Wins: {equity_stats['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {equity_stats['max_consecutive_losses']}")
    
    print(f"\n--- Risk Metrics ---")
    print(f"Daily Starting Capital: $10,000.00 (resets each day)")
    print(f"Cumulative P&L: ${equity_stats['cumulative_pnl']:,.2f}")
    print(f"Peak Cumulative P&L: ${equity_stats['peak_cumulative_pnl']:,.2f}")
    print(f"Max Drawdown (from cumulative peak): ${equity_stats['max_drawdown_from_peak']:,.2f}")
    print(f"Current Drawdown: ${equity_stats['current_drawdown']:,.2f}")
    print(f"Max Intraday DD (single day): ${max_intraday_dd:.2f}")
    
    # Calculate Sharpe-like ratio (annualized)
    if len(equity_stats['daily_pnls']) > 1:
        daily_returns = np.array(equity_stats['daily_pnls']) / 10000  # Returns as fraction of $10K
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        print(f"Sharpe Ratio (annualized): {sharpe:.2f}")
    
    print(f"{'='*80}\n")
    
    return equity_stats
def export_all_trades_single_csv(
    all_episodes_data: List[Tuple[str, TradingEnvironment]],
    output_file: str = "trade_logs/all_test_trades.csv"
):
    """
    Export all trades from multiple episodes into a single CSV file.
    
    Args:
        all_episodes_data: List of (date, env) tuples
        output_file: Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'trade_id',
            'date',
            'entry_bar',
            'exit_bar',
            'direction',
            'entry_price',
            'exit_price',
            'stop_price',
            'lot_size',
            'pnl',
            'mae',
            'duration_minutes',
            'exit_reason',
            'risk_amount',
            'balance_before',
            'balance_after'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        trade_counter = 0
        
        for date, env in all_episodes_data:
            if not env.trades_history:
                continue
            
            # Track balance through the day
            current_balance = 10000.0
            
            for trade in env.trades_history:
                trade_counter += 1
                balance_before = current_balance
                balance_after = current_balance + trade['pnl']
                current_balance = balance_after
                
                row = {
                    'trade_id': trade_counter,
                    'date': date,
                    'entry_bar': trade['entry_bar'],
                    'exit_bar': trade['exit_bar'],
                    'direction': trade['position_type'],
                    'entry_price': f"{trade['entry_price']:.2f}",
                    'exit_price': f"{trade['exit_price']:.2f}",
                    'stop_price': f"{trade['stop_price']:.2f}",
                    'lot_size': f"{trade['lot_size']:.4f}",
                    'pnl': f"{trade['pnl']:.2f}",
                    'mae': f"{trade['mae']:.2f}",
                    'duration_minutes': trade['duration'],
                    'exit_reason': trade['exit_reason'],
                    'risk_amount': f"{trade['risk_amount']:.2f}",
                    'balance_before': f"{balance_before:.2f}",
                    'balance_after': f"{balance_after:.2f}"
                }
                writer.writerow(row)
    
    print(f"\nTrade Export: {trade_counter} trades saved to {output_file}")


def evaluate_episode_with_export(
    env: TradingEnvironment,
    agent: DQNAgent,
    feature_mask: np.ndarray,
    date: str,  # EXPERIMENT-012: Add date parameter
    eval_mode: str = 'argmax',
    temperature: float = 0.15
) -> Tuple[Dict, TradingEnvironment]:
    """
    Run one evaluation episode and return both metrics and environment.
    Environment is returned so we can export trades later.
    
    Args:
        env: Trading environment
        agent: DQN agent
        feature_mask: Feature mask to apply
        date: Date string for this episode
        eval_mode: Evaluation policy mode ('argmax' or 'softmax')
        temperature: Temperature for softmax sampling (only used if eval_mode='softmax')
    
    Returns:
        Tuple of (metrics dict, environment instance)
    """
    state, info = env.reset()
    state = apply_feature_mask(state, feature_mask)
    
    episode_reward = 0.0
    terminated = False
    truncated = False
    
    # EXPERIMENT-009-DIAGNOSTIC: Accumulate reward components
    reward_components_sum = {}
    
    while not (terminated or truncated):
        action_mask = env.get_action_mask()
        action = agent.select_action(state, action_mask, training=False,
                                     eval_mode=eval_mode, temperature=temperature)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = apply_feature_mask(next_state, feature_mask)
        
        # EXPERIMENT-009-DIAGNOSTIC: Accumulate reward components if available
        if 'reward_components' in info:
            for key, value in info['reward_components'].items():
                if key != 'trade_just_closed':  # Skip boolean flags
                    reward_components_sum[key] = reward_components_sum.get(key, 0.0) + value
        
        state = next_state
        episode_reward += reward
    
    # Get episode summary from environment
    summary = env.get_episode_summary()
    
    # Create metrics
    metrics = {
        'date': date,  # EXPERIMENT-012: Include date
        'episode_reward': episode_reward,
        'realized_pnl': summary['total_pnl'],
        'max_drawdown': env.balance - env.peak_balance,
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate'],
        'avg_mae': np.mean([t['mae'] for t in env.trades_history]) if env.trades_history else 0.0,
        'avg_trade_duration': summary['avg_duration'] if summary['total_trades'] > 0 else 0.0,
        'exit_reasons': summary.get('exit_reasons', {}),
        'reward_components_sum': reward_components_sum  # EXPERIMENT-009-DIAGNOSTIC
    }
    
    return metrics, env

def print_phase_summary(phase_name: str, all_metrics: List[Dict]):
    """Print aggregated phase/fold summary with cumulative equity tracking."""
    if not all_metrics:
        return
    
    total_pnl = sum(m['realized_pnl'] for m in all_metrics)
    max_intraday_dd = min(m['max_drawdown'] for m in all_metrics)
    total_trades = sum(m['total_trades'] for m in all_metrics)
    daily_pnls = [m['realized_pnl'] for m in all_metrics]
    
    # Calculate cumulative equity curve drawdown (EVALUATION ONLY)
    equity_stats = calculate_cumulative_equity_drawdown(all_metrics)
    
    intraday_ratio = total_pnl / abs(max_intraday_dd) if max_intraday_dd != 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"{phase_name} SUMMARY")
    print(f"{'='*80}")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Daily PnL Variance: {np.var(daily_pnls):.2f}")
    print(f"\nIntraday Risk (env-level, affects training):")
    print(f"  Max Intraday DD: ${max_intraday_dd:.2f}")
    print(f"  PnL / Intraday DD: {intraday_ratio:.2f}")
    print(f"\nCumulative Equity Risk (evaluation only, does NOT affect training):")
    print(f"  Starting Balance: $10,000.00")
    print(f"  Final Balance: ${equity_stats['running_balance']:,.2f}")
    print(f"  Peak Balance: ${equity_stats['peak_balance']:,.2f}")
    print(f"  Max Cumulative DD: ${equity_stats['max_drawdown']:,.2f} ({equity_stats['max_drawdown_pct']:.2f}%)")
    print(f"  [Note: DD calculation includes intraday peak reconstruction]")
    print(f"{'='*80}\n")


def save_model(
    agent: DQNAgent,
    save_dir: str,
    model_name: str,
    metadata: Dict
):
    """Save model checkpoint with metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    agent.save(model_path)
    
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")


def save_checkpoint(
    agent: DQNAgent,
    save_dir: str,
    checkpoint_name: str,
    day_idx: int,
    date: str
):
    """Save intermediate training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, f"{checkpoint_name}_day{day_idx}_{date}.pt")
    agent.save(checkpoint_path)
    
    print(f"  Checkpoint saved: {checkpoint_path}")


# ============================================================================
# PHASE A: BURN-IN
# ============================================================================

def run_burnin_phase(
    data_loader: DataLoader,
    feature_phase_start: int = 1,
    feature_phase_end: int = 2,
    train_freq: int = 4,
    save_dir: str = "models",
    checkpoint_every: int = 100
):
    """
    Phase A: Burn-in training on 2021-2022.
    Feature progression from phase_start to phase_end.
    Returns the measured average steps per day for epsilon calibration.
    """
    # Setup logging for this phase
    logger = setup_phase_logging("Burn-in")
    
    print("\n" + "="*80)
    print("PHASE A: BURN-IN (2021-2022)")
    print("="*80)
    
    start_date = "2021-01-04"
    end_date = "2022-12-31"
    
    df = data_loader.data
    df = df[(df['Date'].astype(str) >= start_date) & (df['Date'].astype(str) <= end_date)]
    trading_days = sorted(df['Date'].astype(str).unique())
    
    print(f"Training days: {len(trading_days)}")
    print(f"Feature phases: {feature_phase_start} -> {feature_phase_end}")
    print(f"Reward function hash: {get_reward_function_hash()}")
    
    # PHASE 1: Calibration run (first 50 days) to measure steps per day
    # METRICS-ONLY: Measures environment interaction rates
    # Does NOT contribute to training - replay buffer will be discarded
    print(f"\n{'='*80}")
    print("CALIBRATION: Measuring average steps per episode (METRICS-ONLY)")
    print(f"{'='*80}")
    
    calibration_days = min(50, len(trading_days))
    calibration_metrics = []
    
    # Create temporary agent for calibration (will be discarded)
    state_shape = (60, len(ALL_FEATURES))
    temp_agent = DQNAgent(
        state_shape=state_shape,
        n_actions=4,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100_000,  # Temporary value, will be recalculated
        buffer_capacity=250_000,
        batch_size=64,
        target_update_frequency=5_000
    )
    
    feature_mask = create_feature_mask(feature_phase_start)
    
    for day_idx in range(calibration_days):
        date = trading_days[day_idx]
        episode_loader = EpisodeDataLoader(data_loader, date)
        env = TradingEnvironment(episode_loader, window_size=60)
        metrics = train_episode(env, temp_agent, feature_mask, train_freq=train_freq)
        calibration_metrics.append(metrics)
    
    # Calculate observed steps per day
    observed_steps_per_day = np.mean([m['steps'] for m in calibration_metrics])
    
    print(f"\nCalibration complete:")
    print(f"  Days sampled: {calibration_days}")
    print(f"  Observed steps/day: {observed_steps_per_day:.1f}")
    print(f"  Min steps: {min(m['steps'] for m in calibration_metrics)}")
    print(f"  Max steps: {max(m['steps'] for m in calibration_metrics)}")
    print(f"  Std dev: {np.std([m['steps'] for m in calibration_metrics]):.1f}")
    
    # Calculate epsilon decay steps using observed data
    epsilon_decay_steps = calculate_epsilon_decay_steps(len(trading_days), observed_steps_per_day)
    
    # PHASE 2: Full training with calibrated epsilon decay
    print(f"\n{'='*80}")
    print("FULL TRAINING: Using calibrated epsilon decay")
    print(f"{'='*80}")
    
    # Initialize FRESH agent with correct epsilon decay
    # NO replay buffer reuse - strict phase separation
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=4,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_capacity=250_000,
        batch_size=64,
        target_update_frequency=5_000
    )
    
    # Calibration agent is discarded - NO replay buffer transfer
    # Main training starts with fresh weights and empty replay buffer
    
    # Feature progression schedule
    total_phases = feature_phase_end - feature_phase_start + 1
    days_per_phase = len(trading_days) // total_phases
    current_phase = feature_phase_start
    feature_mask = create_feature_mask(current_phase)
    
    print(f"Days per phase: {days_per_phase}")
    
    all_metrics = []
    
    for day_idx, date in enumerate(trading_days):
        # Update feature phase if needed
        phase_boundary = (day_idx // days_per_phase) + feature_phase_start
        if phase_boundary != current_phase and phase_boundary <= feature_phase_end:
            current_phase = phase_boundary
            feature_mask = create_feature_mask(current_phase)
            print(f"\n>>> Feature phase advanced to {current_phase}")
        
        # Create environment for this day
        episode_loader = EpisodeDataLoader(data_loader, date)
        env = TradingEnvironment(episode_loader, window_size=60)
        
        metrics = train_episode(env, agent, feature_mask, train_freq=train_freq)
        all_metrics.append(metrics)
        
        if (day_idx + 1) % 50 == 0 or day_idx == len(trading_days) - 1:
            print(f"\n[{day_idx+1}/{len(trading_days)}] ", end="")
            print_episode_summary(date, metrics)
            agent_stats = agent.get_stats()
            print(f"  Agent: ε={agent_stats['epsilon']:.4f}, "
                  f"buffer={agent_stats['buffer_size']}, "
                  f"steps={agent_stats['training_steps']}")
        
        # Save checkpoint periodically
        if (day_idx + 1) % checkpoint_every == 0:
            save_checkpoint(agent, save_dir, "burnin_checkpoint", day_idx + 1, date)
    
    print_phase_summary("BURN-IN PHASE", all_metrics)
    
    # Calculate cumulative equity stats for metadata (EVALUATION ONLY)
    equity_stats = calculate_cumulative_equity_drawdown(all_metrics)
    
    # Save burn-in model with calibration metadata
    metadata = {
        'phase': 'burn-in',
        'training_window': f"{start_date} to {end_date}",
        'feature_phases': f"{feature_phase_start}-{feature_phase_end}",
        'total_episodes': len(trading_days),
        'total_pnl': sum(m['realized_pnl'] for m in all_metrics),
        'max_intraday_drawdown': min(m['max_drawdown'] for m in all_metrics),
        'total_trades': sum(m['total_trades'] for m in all_metrics),
        'observed_steps_per_day': observed_steps_per_day,  # Store for other phases
        'calibration_days': calibration_days,
        'cumulative_equity': {  # EVALUATION ONLY - does not affect training
            'starting_balance': 10_000.0,
            'final_balance': equity_stats['running_balance'],
            'peak_balance': equity_stats['peak_balance'],
            'max_drawdown': equity_stats['max_drawdown'],
            'max_drawdown_pct': equity_stats['max_drawdown_pct']
        },
        'reward_function_hash': get_reward_function_hash(),
        'hyperparameters': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': epsilon_decay_steps,
            'buffer_capacity': 250_000,
            'batch_size': 64,
            'target_update_frequency': 5_000,
            'train_freq': train_freq
        },
        'agent_stats': agent.get_stats()
    }
    save_model(agent, save_dir, "burnin_agent", metadata)
    
    # Restore logging
    restore_logging(logger)
    
    return agent, observed_steps_per_day


# ============================================================================
# PHASE B: WALK-FORWARD VALIDATION
# ============================================================================

def run_walkforward_validation(
    data_loader: DataLoader,
    observed_steps_per_day: float,
    feature_phase: int = 2,
    train_freq: int = 4,
    save_dir: str = "models",
    checkpoint_every: int = 100
):
    """
    Phase B: Expanding walk-forward validation.
    Reset agent from scratch for each fold.
    
    Args:
        data_loader: DataLoader instance
        observed_steps_per_day: Average steps per episode measured from burn-in
        feature_phase: Feature phase to use
        train_freq: Training frequency
        save_dir: Directory to save models
        checkpoint_every: Save checkpoint every N days
    """
    # Setup logging for walk-forward phase
    logger = setup_phase_logging("Walk-Forward")
    
    print("\n" + "="*80)
    print("PHASE B: WALK-FORWARD VALIDATION")
    print("="*80)
    print(f"Using observed steps/day: {observed_steps_per_day:.1f} (from burn-in)")
    print(f"Reward function hash: {get_reward_function_hash()}")
    
    folds = [
        {
            'name': 'WF-1',
            'train_start': '2021-01-04',
            'train_end': '2022-12-31',
            'val_start': '2023-01-01',
            'val_end': '2023-03-31'
        },
        {
            'name': 'WF-2',
            'train_start': '2021-01-04',
            'train_end': '2023-03-31',
            'val_start': '2023-04-01',
            'val_end': '2023-06-30'
        },
        {
            'name': 'WF-3',
            'train_start': '2021-01-04',
            'train_end': '2023-06-30',
            'val_start': '2023-07-01',
            'val_end': '2023-09-30'
        },
        {
            'name': 'WF-4',
            'train_start': '2021-01-04',
            'train_end': '2023-09-30',
            'val_start': '2023-10-01',
            'val_end': '2023-12-31'
        }
    ]
    
    feature_mask = create_feature_mask(feature_phase)
    state_shape = (60, len(ALL_FEATURES))
    
    fold_results = []
    
    for fold in folds:
        print(f"\n{'='*80}")
        print(f"FOLD: {fold['name']}")
        print(f"Train: {fold['train_start']} -> {fold['train_end']}")
        print(f"Val:   {fold['val_start']} -> {fold['val_end']}")
        print(f"{'='*80}")
        
        # Get training days for epsilon calculation
        df = data_loader.data
        train_df = df[(df['Date'].astype(str) >= fold['train_start']) & 
                      (df['Date'].astype(str) <= fold['train_end'])]
        train_days = sorted(train_df['Date'].astype(str).unique())
        
        # Calculate epsilon decay steps using observed steps per day
        epsilon_decay_steps = calculate_epsilon_decay_steps(len(train_days), observed_steps_per_day)
        
        # Reset agent from scratch
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=4,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=epsilon_decay_steps,
            buffer_capacity=250_000,
            batch_size=64,
            target_update_frequency=5_000
        )
        
        # Training phase
        print(f"\n--- Training {fold['name']} ---")
        
        train_metrics = []
        for day_idx, date in enumerate(train_days):
            episode_loader = EpisodeDataLoader(data_loader, date)
            env = TradingEnvironment(episode_loader, window_size=60)
            metrics = train_episode(env, agent, feature_mask, train_freq=train_freq)
            train_metrics.append(metrics)
            
            if (day_idx + 1) % 100 == 0 or day_idx == len(train_days) - 1:
                print(f"[{day_idx+1}/{len(train_days)}] Date: {date}, "
                      f"PnL: ${metrics['realized_pnl']:.2f}, "
                      f"ε: {agent.get_stats()['epsilon']:.4f}")
            
            # Save checkpoint periodically
            if (day_idx + 1) % checkpoint_every == 0:
                save_checkpoint(agent, save_dir, f"wf_{fold['name'].lower()}_checkpoint", 
                              day_idx + 1, date)
        
        print_phase_summary(f"{fold['name']} Training", train_metrics)
        
        # Validation phase
        print(f"\n--- Validation {fold['name']} ---")
        val_df = df[(df['Date'].astype(str) >= fold['val_start']) & 
                    (df['Date'].astype(str) <= fold['val_end'])]
        val_days = sorted(val_df['Date'].astype(str).unique())
        
        val_metrics = []
        for date in val_days:
            episode_loader = EpisodeDataLoader(data_loader, date)
            env = TradingEnvironment(episode_loader, window_size=60)
            metrics = evaluate_episode(env, agent, feature_mask)
            val_metrics.append(metrics)
            print_episode_summary(date, metrics, prefix="  ")
        
        print_phase_summary(f"{fold['name']} Validation", val_metrics)
        
        # Calculate cumulative equity stats for both train and validation (EVALUATION ONLY)
        train_equity_stats = calculate_cumulative_equity_drawdown(train_metrics)
        val_equity_stats = calculate_cumulative_equity_drawdown(val_metrics)
        
        # Save fold model
        fold_metadata = {
            'fold': fold['name'],
            'train_window': f"{fold['train_start']} to {fold['train_end']}",
            'val_window': f"{fold['val_start']} to {fold['val_end']}",
            'feature_phase': feature_phase,
            'active_features': get_active_features(feature_phase),
            'train_days': len(train_days),
            'val_days': len(val_days),
            'train_pnl': sum(m['realized_pnl'] for m in train_metrics),
            'train_max_intraday_dd': min(m['max_drawdown'] for m in train_metrics),
            'val_pnl': sum(m['realized_pnl'] for m in val_metrics),
            'val_max_intraday_dd': min(m['max_drawdown'] for m in val_metrics),
            'val_total_trades': sum(m['total_trades'] for m in val_metrics),
            'val_win_rate': np.mean([m['win_rate'] for m in val_metrics if m['total_trades'] > 0]),
            'train_cumulative_equity': {  # EVALUATION ONLY - does not affect training
                'starting_balance': 10_000.0,
                'final_balance': train_equity_stats['running_balance'],
                'peak_balance': train_equity_stats['peak_balance'],
                'max_drawdown': train_equity_stats['max_drawdown'],
                'max_drawdown_pct': train_equity_stats['max_drawdown_pct']
            },
            'val_cumulative_equity': {  # EVALUATION ONLY - does not affect training
                'starting_balance': 10_000.0,
                'final_balance': val_equity_stats['running_balance'],
                'peak_balance': val_equity_stats['peak_balance'],
                'max_drawdown': val_equity_stats['max_drawdown'],
                'max_drawdown_pct': val_equity_stats['max_drawdown_pct']
            },
            'observed_steps_per_day': observed_steps_per_day,  # Store for reference
            'reward_function_hash': get_reward_function_hash(),
            'hyperparameters': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay_steps': epsilon_decay_steps,
                'buffer_capacity': 250_000,
                'batch_size': 64,
                'target_update_frequency': 5_000,
                'train_freq': train_freq
            },
            'agent_stats': agent.get_stats()
        }
        save_model(agent, save_dir, f"wf_{fold['name'].lower()}_agent", fold_metadata)
        
        fold_results.append({
            'fold': fold['name'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'metadata': fold_metadata
        })
    
    # Overall walk-forward summary
    print("\n" + "="*80)
    print("WALK-FORWARD OVERALL SUMMARY")
    print("="*80)
    for result in fold_results:
        print(f"\n{result['fold']}:")
        print(f"  Val PnL: ${result['metadata']['val_pnl']:.2f}")
        print(f"  Val Intraday DD: ${result['metadata']['val_max_intraday_dd']:.2f}")
        print(f"  Val Cumulative DD: ${result['metadata']['val_cumulative_equity']['max_drawdown']:.2f} "
              f"({result['metadata']['val_cumulative_equity']['max_drawdown_pct']:.2f}%)")
        print(f"  Val Days: {result['metadata']['val_days']}")
        print(f"  Val Trades: {result['metadata']['val_total_trades']}")
        print(f"  Val Win Rate: {result['metadata']['val_win_rate']:.2%}")
    
    # Restore logging
    restore_logging(logger)
    
    return fold_results


# ============================================================================
# EXPERIMENT REPORTING HELPERS
# ============================================================================

def calculate_daily_dd_breaches(all_metrics: List[Dict], dd_threshold: float = -500.0) -> int:
    """
    Calculate number of days that breached the daily drawdown threshold.
    
    Args:
        all_metrics: List of episode metrics containing 'max_drawdown'
        dd_threshold: Drawdown threshold in dollars (e.g., -500 for 5% of $10k)
        
    Returns:
        Number of days with max_drawdown <= dd_threshold
    """
    if not all_metrics:
        return 0
    
    breaches = sum(1 for m in all_metrics if m['max_drawdown'] <= dd_threshold)
    return breaches


def print_results_packet(
    experiment_id: str,
    train_metrics: List[Dict],
    test_metrics: List[Dict],
    train_equity_stats: Dict,
    test_equity_stats: Dict,
    agent_stats: Dict,
    epsilon_decay_steps: int,
    eval_mode: str = 'argmax',
    temperature: float = 0.15,
    seeds: Optional[List[int]] = None,
    notes: str = ""
):
    """
    Print standardized RESULTS PACKET for experiment reporting.
    
    Args:
        experiment_id: Experiment identifier (e.g., "EXPERIMENT-001")
        train_metrics: List of training episode metrics
        test_metrics: List of test episode metrics
        train_equity_stats: Training equity statistics from calculate_daily_equity_curve
        test_equity_stats: Test equity statistics from calculate_daily_equity_curve
        agent_stats: Agent statistics (training_steps, epsilon, buffer_size)
        epsilon_decay_steps: Total epsilon decay steps used
        eval_mode: Evaluation mode used ('argmax' or 'softmax')
        temperature: Temperature used if eval_mode='softmax'
        seeds: List of random seeds used (or None if not set)
        notes: Additional notes about the run
    """
    # Calculate daily DD breaches (5% of $10k = $500)
    train_dd_breaches = calculate_daily_dd_breaches(train_metrics, dd_threshold=-500.0)
    test_dd_breaches = calculate_daily_dd_breaches(test_metrics, dd_threshold=-500.0)
    
    # Build eval protocol description
    if eval_mode == 'argmax':
        eval_protocol = "Full-period daily episodes over Train and Test windows, deterministic argmax policy, epsilon=0"
    else:
        eval_protocol = f"Full-period daily episodes over Train and Test windows, stochastic softmax policy (temperature={temperature}), epsilon=0"
    
    # Build seeds string
    if seeds is None:
        seeds_str = "[not set]"
    else:
        seeds_str = str(seeds)
    
    # Calculate trades per day
    train_trades_per_day = sum(m['total_trades'] for m in train_metrics) / len(train_metrics) if train_metrics else 0.0
    test_trades_per_day = sum(m['total_trades'] for m in test_metrics) / len(test_metrics) if test_metrics else 0.0
    
    # Calculate Sharpe ratios
    if len(train_metrics) > 1:
        train_daily_returns = np.array([m['realized_pnl'] for m in train_metrics]) / 10000
        train_sharpe = np.mean(train_daily_returns) / np.std(train_daily_returns) * np.sqrt(252)
    else:
        train_sharpe = 0.0
    
    if len(test_metrics) > 1:
        test_daily_returns = np.array([m['realized_pnl'] for m in test_metrics]) / 10000
        test_sharpe = np.mean(test_daily_returns) / np.std(test_daily_returns) * np.sqrt(252)
    else:
        test_sharpe = 0.0
    
    # Print RESULTS PACKET
    print("\n" + "="*80)
    print("RESULTS PACKET")
    print("="*80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Training timesteps completed: {agent_stats.get('training_steps', 'N/A')}")
    print(f"Seeds: {seeds_str}")
    print(f"Eval protocol: {eval_protocol}")
    print()
    print("Key metrics:")
    print()
    
    # Training metrics
    train_win_rate = np.mean([m['win_rate'] for m in train_metrics if m['total_trades'] > 0]) if train_metrics else 0.0
    print(f"Train:")
    print(f"  Win rate: {train_win_rate:.2%}")
    print(f"  PnL: ${train_equity_stats.get('cumulative_pnl', 0.0):,.2f}")
    print(f"  Sharpe: {train_sharpe:.2f}")
    print(f"  Profit factor: {train_equity_stats.get('profit_factor', 0.0):.2f}")
    print(f"  Daily DD breaches: {train_dd_breaches}/{len(train_metrics)} days")
    print(f"  Monthly DD (from peak): ${train_equity_stats.get('max_drawdown_from_peak', 0.0):,.2f}")
    print(f"  Max DD (single day intraday): ${min(m['max_drawdown'] for m in train_metrics) if train_metrics else 0.0:.2f}")
    print(f"  Trades/day: {train_trades_per_day:.1f}")
    print()
    
    # Test metrics
    test_win_rate = np.mean([m['win_rate'] for m in test_metrics if m['total_trades'] > 0]) if test_metrics else 0.0
    print(f"Test:")
    print(f"  Win rate: {test_win_rate:.2%}")
    print(f"  PnL: ${test_equity_stats.get('cumulative_pnl', 0.0):,.2f}")
    print(f"  Sharpe: {test_sharpe:.2f}")
    print(f"  Profit factor: {test_equity_stats.get('profit_factor', 0.0):.2f}")
    print(f"  Daily DD breaches: {test_dd_breaches}/{len(test_metrics)} days")
    print(f"  Monthly DD (from peak): ${test_equity_stats.get('max_drawdown_from_peak', 0.0):,.2f}")
    print(f"  Max DD (single day intraday): ${min(m['max_drawdown'] for m in test_metrics) if test_metrics else 0.0:.2f}")
    print(f"  Trades/day: {test_trades_per_day:.1f}")
    print()
    
    # Additional eval metrics (not applicable for full-period evaluation)
    print(f"Best eval mean return: N/A (full-period evaluation)")
    print(f"Last eval mean return: N/A (full-period evaluation)")
    print(f"Eval success rate: N/A (full-period evaluation)")
    print(f"Mean episode length (eval): {np.mean([sum(1 for _ in range(len(train_metrics))) for _ in [train_metrics]]) if train_metrics else 0:.1f} bars/episode")
    print()
    
    # Stability notes
    print(f"Stability notes: {notes if notes else 'No issues observed'}")
    print()
    print("Notes:")
    print(f"  - Evaluation uses {eval_mode} policy")
    if eval_mode == 'softmax':
        print(f"  - Softmax temperature: {temperature}")
    print(f"  - Epsilon decay steps: {epsilon_decay_steps:,}")
    print(f"  - Current epsilon: {agent_stats.get('epsilon', 'N/A')}")
    print(f"  - Buffer size: {agent_stats.get('buffer_size', 'N/A'):,}")
    print(f"  - Env steps: {agent_stats.get('env_steps', 'N/A'):,}")  # EXPERIMENT-006
    print(f"  - Update steps: {agent_stats.get('update_steps', 'N/A'):,}")  # EXPERIMENT-006
    print(f"  - Reward config: MAX_TRADES_PER_DAY={TradingEnvironment.MAX_TRADES_PER_DAY}, "
          f"MIN_TRADES_TARGET={TradingEnvironment.MIN_TRADES_TARGET}, "
          f"EXECUTION_PENALTY_R={TradingEnvironment.EXECUTION_PENALTY_R}, "
          f"ACTION_PENALTY_R={TradingEnvironment.ACTION_PENALTY_R}, "
          f"STOPLOSS_EVENT_PENALTY_R={TradingEnvironment.STOPLOSS_EVENT_PENALTY_R}")
    
    # Calculate days hitting trade cap
    train_days_at_cap = sum(1 for m in train_metrics if m.get('total_trades', 0) >= 10)
    test_days_at_cap = sum(1 for m in test_metrics if m.get('total_trades', 0) >= 10)
    print(f"  - Days hit trade cap: {train_days_at_cap}/{len(train_metrics)} (Train), {test_days_at_cap}/{len(test_metrics)} (Test)")
    print("="*80 + "\n")


# ============================================================================
# PHASE C: FINAL TRAIN & TEST
# ============================================================================

def run_final_train_test(
    data_loader: DataLoader,
    observed_steps_per_day: float,
    feature_phase: int = 4,
    train_freq: int = 4,
    save_dir: str = "models",
    checkpoint_every: int = 100,
    eval_mode: str = 'argmax',
    eval_temperature: float = 0.15
):
    """
    Phase C: Final training on 2021-2024, test on 2025.
    
    Args:
        data_loader: DataLoader instance
        observed_steps_per_day: Average steps per episode measured from burn-in
        feature_phase: Feature phase to use
        train_freq: Training frequency
        save_dir: Directory to save models
        checkpoint_every: Save checkpoint every N days
        eval_mode: Evaluation policy mode ('argmax' or 'softmax')
        eval_temperature: Temperature for softmax sampling (only used if eval_mode='softmax')
    """
    # Setup logging for final train/test phase
    logger = setup_phase_logging("Final-Train-Test")
    
    # EXPERIMENT-014: Experiment ID constant
    EXPERIMENT_ID = "EXPERIMENT-015"
    
    print("\n" + "="*80)
    print("PHASE C: FINAL TRAIN & TEST")
    print("="*80)
    print(f"Using observed steps/day: {observed_steps_per_day:.1f} (from burn-in)")
    print(f"Reward function hash: {get_reward_function_hash()}")
    # EXPERIMENT-014: Print mechanics hash (separate from reward hash)
    print(f"Mechanics hash (EXP-014): {get_mechanics_hash()}")
    print(f"Evaluation mode: {eval_mode}" + (f" (temperature={eval_temperature})" if eval_mode == 'softmax' else ""))
    # EXPERIMENT-011: Print current config dynamically from TradingEnvironment constants
    print(f"{EXPERIMENT_ID} config:")
    print(f"  MAX_TRADES_PER_DAY={TradingEnvironment.MAX_TRADES_PER_DAY}")
    print(f"  BUDGET_LAMBDA={TradingEnvironment.BUDGET_LAMBDA}")
    print(f"  MIN_TRADES_TARGET={TradingEnvironment.MIN_TRADES_TARGET}")
    print(f"  UNUSED_LAMBDA={TradingEnvironment.UNUSED_LAMBDA}")
    print(f"  EXECUTION_PENALTY_R={TradingEnvironment.EXECUTION_PENALTY_R}")
    print(f"  ACTION_PENALTY_R={TradingEnvironment.ACTION_PENALTY_R}")
    print(f"  STOPLOSS_EVENT_PENALTY_R={TradingEnvironment.STOPLOSS_EVENT_PENALTY_R}")
    print(f"  REVERSAL_EVENT_PENALTY_R={TradingEnvironment.REVERSAL_EVENT_PENALTY_R}")  # EXPERIMENT-011
    print(f"  STOPLOSS_COOLDOWN_THRESHOLD={TradingEnvironment.STOPLOSS_COOLDOWN_THRESHOLD}")  # EXPERIMENT-013
    # EXPERIMENT-014: Print session allocation + soft gate constants
    print(f"  RESERVED_ENTRIES_LONDON={TradingEnvironment.RESERVED_ENTRIES_LONDON}")  # EXPERIMENT-014
    print(f"  RESERVED_ENTRIES_NY={TradingEnvironment.RESERVED_ENTRIES_NY}")  # EXPERIMENT-014
    print(f"  STOPLOSS_SOFT_GATE_THRESHOLD={TradingEnvironment.STOPLOSS_SOFT_GATE_THRESHOLD}")  # EXPERIMENT-014
    print(f"  ATR_MULTIPLIER={TradingEnvironment.ATR_MULTIPLIER}")
    print(f"  MANUAL_EXIT_AT_OPEN={TradingEnvironment.MANUAL_EXIT_AT_OPEN}")
    print(f"  INACTIVITY_BASELINE_REWARD={TradingEnvironment.INACTIVITY_BASELINE_REWARD}")  # EXPERIMENT-010
    print(f"  PATIENCE_REWARD={TradingEnvironment.PATIENCE_REWARD}")  # EXPERIMENT-010
    print(f"  MIN_HOLD_BARS={TradingEnvironment.MIN_HOLD_BARS} (now enforced on reversals)")  # EXPERIMENT-011
    print(f"  QR-DQN: n_quantiles=51, risk_fraction=0.25, cql_enabled=False")
    
    train_start = "2021-01-04"
    train_end = "2024-12-31"
    test_start = "2025-01-02"
    test_end = "2025-12-31"
    
    feature_mask = create_feature_mask(feature_phase)
    state_shape = (60, len(ALL_FEATURES))
    
    # Get training days for epsilon calculation
    df = data_loader.data
    train_df = df[(df['Date'].astype(str) >= train_start) & (df['Date'].astype(str) <= train_end)]
    train_days = sorted(train_df['Date'].astype(str).unique())
    
    # Calculate epsilon decay steps using observed steps per day
    epsilon_decay_steps = calculate_epsilon_decay_steps(len(train_days), observed_steps_per_day)
    
    # Initialize fresh agent with QR-DQN (EXPERIMENT-008)
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=4,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_capacity=250_000,
        batch_size=64,
        target_update_frequency=5_000,
        cql_alpha=0.2,  # Not used when cql_enabled=False
        cql_temperature=1.0,  # Not used when cql_enabled=False
        cql_enabled=False,  # EXPERIMENT-008: Disable CQL
        n_quantiles=51,  # EXPERIMENT-008: Number of quantiles
        quantile_huber_kappa=1.0,  # EXPERIMENT-008: Huber threshold
        risk_fraction=0.25  # EXPERIMENT-008: Bottom 25% quantiles for risk-averse selection
    )
    
    # Training phase
    print(f"\n--- Final Training: {train_start} -> {train_end} ---")
    
    print(f"Training days: {len(train_days)}")
    print(f"Feature phase: {feature_phase} (all features)")
    
    train_metrics = []
    for day_idx, date in enumerate(train_days):
        episode_loader = EpisodeDataLoader(data_loader, date)
        env = TradingEnvironment(episode_loader, window_size=60)
        metrics = train_episode(env, agent, feature_mask, date, train_freq=train_freq)  # EXPERIMENT-012: Pass date
        train_metrics.append(metrics)
        
        if (day_idx + 1) % 100 == 0 or day_idx == len(train_days) - 1:
            agent_stats = agent.get_stats()
            # EXPERIMENT-006: Log both env_steps and update_steps
            print(f"[{day_idx+1}/{len(train_days)}] Date: {date}, "
                  f"PnL: ${metrics['realized_pnl']:.2f}, "
                  f"ε: {agent_stats['epsilon']:.4f}, "
                  f"env_steps: {agent_stats['env_steps']:,}, "
                  f"update_steps: {agent_stats['update_steps']:,}")
        
        # Save checkpoint periodically
        if (day_idx + 1) % checkpoint_every == 0:
            save_checkpoint(agent, save_dir, "final_checkpoint", day_idx + 1, date)
    
    train_equity_stats_daily = print_daily_performance_summary("Final Training", train_metrics)
    
    # Test phase with trade export
    print(f"\n--- Final Test: {test_start} -> {test_end} ---")
    test_df = df[(df['Date'].astype(str) >= test_start) & (df['Date'].astype(str) <= test_end)]
    test_days = sorted(test_df['Date'].astype(str).unique())
    
    print(f"Test days: {len(test_days)}")
    
    test_metrics = []
    episodes_with_envs = []  # Store (date, env) for CSV export
    
    for date in test_days:
        episode_loader = EpisodeDataLoader(data_loader, date)
        env = TradingEnvironment(episode_loader, window_size=60)
        metrics, final_env = evaluate_episode_with_export(env, agent, feature_mask, date,  # EXPERIMENT-012: Pass date
                                                          eval_mode=eval_mode, 
                                                          temperature=eval_temperature)
        test_metrics.append(metrics)
        episodes_with_envs.append((date, final_env))
        print_episode_summary(date, metrics, prefix="  ")
    
    # Export all test trades to CSV
    trade_log_dir = os.path.join(save_dir, "trade_logs")
    export_all_trades_single_csv(
        episodes_with_envs, 
        os.path.join(trade_log_dir, "all_test_trades.csv")
    )
    
    # Print improved daily performance summary
    test_equity_stats = print_daily_performance_summary("Final Test", test_metrics)
    
    # EXPERIMENT-009-DIAGNOSTIC: Additional TXT-only diagnostics for FTMO monthly DD and reward dominance analysis
    print("\n" + "="*80)
    print("EXPERIMENT-009 DIAGNOSTICS - TEST 2025 CALENDAR-MONTH DRAWDOWN")
    print("="*80)
    
    # Calculate calendar-month DD table for Test
    # Group test metrics by month
    from collections import defaultdict
    test_df = data_loader.data
    test_df_filtered = test_df[(test_df['Date'].astype(str) >= test_start) & (test_df['Date'].astype(str) <= test_end)]
    test_days_sorted = sorted(test_df_filtered['Date'].astype(str).unique())
    
    # Build monthly aggregation
    monthly_data = defaultdict(lambda: {
        'dates': [],
        'daily_pnls': [],
        'daily_intraday_dds': [],
        'total_trades': 0,
        'stop_loss_exits': 0,
        'manual_exits': 0,
        'cap_hit_days': 0
    })
    
    for i, date in enumerate(test_days_sorted):
        month_key = date[:7]  # YYYY-MM
        metrics = test_metrics[i]
        monthly_data[month_key]['dates'].append(date)
        monthly_data[month_key]['daily_pnls'].append(metrics['realized_pnl'])
        monthly_data[month_key]['daily_intraday_dds'].append(metrics['max_drawdown'])
        monthly_data[month_key]['total_trades'] += metrics['total_trades']
        
        # Extract exit reasons
        exit_reasons = metrics.get('exit_reasons', {})
        monthly_data[month_key]['stop_loss_exits'] += exit_reasons.get('stop_loss', 0)
        monthly_data[month_key]['manual_exits'] += exit_reasons.get('manual', 0) + exit_reasons.get('reversal', 0)
        
        # Count cap-hit days (>= 10 trades)
        if metrics['total_trades'] >= 10:
            monthly_data[month_key]['cap_hit_days'] += 1
    
    # Print calendar-month table
    print(f"\n{'Month':<10} {'PnL':>12} {'Peak-DD':>12} {'Worst Day':>12} {'Worst Intra':>14} {'Trades':>8} {'StopLoss':>10} {'Manual':>8} {'Cap-Hit':>9}")
    print("-" * 108)
    
    for month in sorted(monthly_data.keys()):
        mdata = monthly_data[month]
        daily_pnls = mdata['daily_pnls']
        
        # Calculate cumulative PnL curve for the month
        cumulative_pnl = np.cumsum(daily_pnls)
        peak_pnl = np.maximum.accumulate(np.insert(cumulative_pnl, 0, 0.0))  # Include 0 as starting point
        drawdowns = cumulative_pnl - peak_pnl[:-1]
        max_dd_from_peak = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        total_pnl = sum(daily_pnls)
        worst_day_pnl = min(daily_pnls)
        worst_intraday_dd = min(mdata['daily_intraday_dds'])
        
        print(f"{month:<10} ${total_pnl:>11.2f} ${max_dd_from_peak:>11.2f} ${worst_day_pnl:>11.2f} ${worst_intraday_dd:>13.2f} "
              f"{mdata['total_trades']:>8} {mdata['stop_loss_exits']:>10} {mdata['manual_exits']:>8} {mdata['cap_hit_days']:>9}")
    
    print("="*80)
    
    # EXPERIMENT-009-DIAGNOSTIC: Top-10 worst Test days by realized PnL
    print("\n" + "="*80)
    print("EXPERIMENT-009 DIAGNOSTICS - TOP-10 WORST TEST DAYS (by PnL)")
    print("="*80)
    
    # Create list of (date, metrics) and sort by realized_pnl ascending
    day_tuples = [(test_days_sorted[i], test_metrics[i]) for i in range(len(test_days_sorted))]
    day_tuples_sorted = sorted(day_tuples, key=lambda x: x[1]['realized_pnl'])
    
    print(f"\n{'Date':<12} {'PnL':>12} {'Intra DD':>12} {'Trades':>8} {'StopLoss':>10} {'Manual':>8} {'Avg MAE':>10} {'Avg Dur':>10}")
    print("-" * 96)
    
    for i, (date, metrics) in enumerate(day_tuples_sorted[:10]):
        exit_reasons = metrics.get('exit_reasons', {})
        stop_loss_count = exit_reasons.get('stop_loss', 0)
        manual_count = exit_reasons.get('manual', 0) + exit_reasons.get('reversal', 0)
        
        print(f"{date:<12} ${metrics['realized_pnl']:>11.2f} ${metrics['max_drawdown']:>11.2f} "
              f"{metrics['total_trades']:>8} {stop_loss_count:>10} {manual_count:>8} "
              f"${metrics['avg_mae']:>9.2f} {metrics['avg_trade_duration']:>10.1f}")
    
    print("="*80)
    
    # EXPERIMENT-009-DIAGNOSTIC: Reward component aggregates for Train and Test
    print("\n" + "="*80)
    print("EXPERIMENT-009 DIAGNOSTICS - REWARD COMPONENT AGGREGATES")
    print("="*80)
    
    # Aggregate reward components for Test
    test_reward_agg = defaultdict(float)
    test_reward_count = 0
    for metrics in test_metrics:
        if 'reward_components_sum' in metrics:
            for key, value in metrics['reward_components_sum'].items():
                test_reward_agg[key] += value
            test_reward_count += 1
    
    # EXPERIMENT-013: Aggregate reward components for Train
    train_reward_agg = defaultdict(float)
    train_reward_count = 0
    for metrics in train_metrics:
        if 'reward_components_sum' in metrics:
            for key, value in metrics['reward_components_sum'].items():
                train_reward_agg[key] += value
            train_reward_count += 1
    
    print("\n--- Train 2021-2024 Reward Components ---")
    print(f"{'Component':<30} {'Sum':>15} {'Mean/Episode':>15}")
    print("-" * 60)
    for key in sorted(train_reward_agg.keys()):
        total = train_reward_agg[key]
        mean = total / train_reward_count if train_reward_count > 0 else 0.0
        print(f"{key:<30} {total:>15.2f} {mean:>15.4f}")
    
    print("\n--- Test 2025 Reward Components ---")
    print(f"{'Component':<30} {'Sum':>15} {'Mean/Episode':>15}")
    print("-" * 60)
    for key in sorted(test_reward_agg.keys()):
        total = test_reward_agg[key]
        mean = total / test_reward_count if test_reward_count > 0 else 0.0
        print(f"{key:<30} {total:>15.2f} {mean:>15.4f}")
    
    # EXPERIMENT-013: Train vs Test reward components comparison (Mean/Episode)
    print("\n--- Train vs Test Reward Components (Mean/Episode) ---")
    print(f"{'Component':<30} {'Train':>15} {'Test':>15} {'Delta':>15}")
    print("-" * 75)
    all_keys = sorted(set(train_reward_agg.keys()) | set(test_reward_agg.keys()))
    for key in all_keys:
        train_mean = train_reward_agg[key] / train_reward_count if train_reward_count > 0 and key in train_reward_agg else 0.0
        test_mean = test_reward_agg[key] / test_reward_count if test_reward_count > 0 and key in test_reward_agg else 0.0
        delta = test_mean - train_mean
        print(f"{key:<30} {train_mean:>15.4f} {test_mean:>15.4f} {delta:>15.4f}")
    
    print("="*80)
    
    # EXPERIMENT-010-DIAGNOSTIC: Additional TXT-only diagnostics for behavior analysis
    print("\n" + "="*80)
    print("EXPERIMENT-010 DIAGNOSTICS - TRADE PATTERNS & BEHAVIOR (TEST 2025)")
    print("="*80)
    
    # B. Trades/day histogram on Test
    print("\n--- Test Trades/Day Histogram ---")
    trades_per_day_list = [m['total_trades'] for m in test_metrics]
    
    buckets = {
        '0': 0,
        '1-2': 0,
        '3-5': 0,
        '6-8': 0,
        '9-10': 0,
        '11+': 0
    }
    
    for trades in trades_per_day_list:
        if trades == 0:
            buckets['0'] += 1
        elif 1 <= trades <= 2:
            buckets['1-2'] += 1
        elif 3 <= trades <= 5:
            buckets['3-5'] += 1
        elif 6 <= trades <= 8:
            buckets['6-8'] += 1
        elif 9 <= trades <= 10:
            buckets['9-10'] += 1
        else:
            buckets['11+'] += 1
    
    total_days = len(test_metrics)
    print(f"{'Trades/Day':<12} {'Count':>8} {'Percentage':>12}")
    print("-" * 32)
    for bucket, count in buckets.items():
        pct = (count / total_days * 100) if total_days > 0 else 0.0
        print(f"{bucket:<12} {count:>8} {pct:>11.1f}%")
    print(f"\nMean trades/day: {np.mean(trades_per_day_list):.2f}")
    print(f"Median trades/day: {np.median(trades_per_day_list):.1f}")
    
    # C. Trade duration percentiles overall and by exit_reason
    print("\n--- Trade Duration Analysis (Test) ---")
    
    # Aggregate all trades from all test episodes
    all_test_trades = []
    for date, env in episodes_with_envs:
        all_test_trades.extend(env.trades_history)
    
    if len(all_test_trades) > 0:
        # Overall duration stats
        durations = [t['duration'] for t in all_test_trades]
        print(f"\nOverall Trade Duration (bars):")
        print(f"  Total trades: {len(all_test_trades)}")
        print(f"  p10: {np.percentile(durations, 10):.1f}")
        print(f"  p50 (median): {np.percentile(durations, 50):.1f}")
        print(f"  p90: {np.percentile(durations, 90):.1f}")
        print(f"  mean: {np.mean(durations):.1f}")
        
        # Duration by exit_reason
        exit_reason_durations = defaultdict(list)
        for t in all_test_trades:
            exit_reason_durations[t['exit_reason']].append(t['duration'])
        
        print(f"\nDuration by Exit Reason:")
        print(f"{'Reason':<15} {'Count':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'Mean':>8}")
        print("-" * 63)
        for reason in sorted(exit_reason_durations.keys()):
            reason_durs = exit_reason_durations[reason]
            if len(reason_durs) > 0:
                print(f"{reason:<15} {len(reason_durs):>8} "
                      f"{np.percentile(reason_durs, 10):>8.1f} "
                      f"{np.percentile(reason_durs, 50):>8.1f} "
                      f"{np.percentile(reason_durs, 90):>8.1f} "
                      f"{np.mean(reason_durs):>8.1f}")
    else:
        print("  No trades found in test set.")
    
    # D. Trade-level win rate on Test
    print("\n--- Trade-Level Win Rate (Test) ---")
    if len(all_test_trades) > 0:
        winning_trades = sum(1 for t in all_test_trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in all_test_trades if t['pnl'] < 0)
        breakeven_trades = sum(1 for t in all_test_trades if t['pnl'] == 0)
        total_trades = len(all_test_trades)
        trade_win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        print(f"  Total trades: {total_trades}")
        print(f"  Winning trades: {winning_trades}")
        print(f"  Losing trades: {losing_trades}")
        print(f"  Breakeven trades: {breakeven_trades}")
        print(f"  Trade-level win rate: {trade_win_rate:.2%}")
        
        # Additional trade stats
        winning_pnls = [t['pnl'] for t in all_test_trades if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in all_test_trades if t['pnl'] < 0]
        
        if len(winning_pnls) > 0:
            print(f"  Avg win: ${np.mean(winning_pnls):.2f}")
        if len(losing_pnls) > 0:
            print(f"  Avg loss: ${np.mean(losing_pnls):.2f}")
        if len(winning_pnls) > 0 and len(losing_pnls) > 0:
            win_loss_ratio = abs(np.mean(winning_pnls) / np.mean(losing_pnls))
            print(f"  Win/Loss ratio: {win_loss_ratio:.2f}")
    else:
        print("  No trades found in test set.")
    
    print("="*80)
    
    # EXPERIMENT-011 DIAGNOSTICS - REGIME & STOP INTENSITY SLICES (TEST 2025)
    print("\n" + "="*80)
    print("EXPERIMENT-011 DIAGNOSTICS - REGIME & STOP INTENSITY SLICES (TEST 2025)")
    print("="*80)
    
    # A. Stop-loss exits/day histogram
    print("\n--- Stop-Loss Intensity Histogram (Test) ---")
    print("Days bucketed by number of stop-loss exits per day")
    
    # EXPERIMENT-012: Build date-to-metrics map for easier lookup
    date_to_metrics = {m['date']: m for m in test_metrics}
    
    # Count stop-loss exits per day
    stoploss_per_day = []
    
    for date, env in episodes_with_envs:
        stoploss_count = sum(1 for t in env.trades_history if t['exit_reason'] == 'stop_loss')
        stoploss_per_day.append((date, stoploss_count))
    
    # Bucket by stop-loss count
    sl_buckets = {
        '0': [],
        '1-2': [],
        '3-5': [],
        '6-8': [],
        '9-10+': []
    }
    
    for date, sl_count in stoploss_per_day:
        if sl_count == 0:
            sl_buckets['0'].append(date)
        elif 1 <= sl_count <= 2:
            sl_buckets['1-2'].append(date)
        elif 3 <= sl_count <= 5:
            sl_buckets['3-5'].append(date)
        elif 6 <= sl_count <= 8:
            sl_buckets['6-8'].append(date)
        else:
            sl_buckets['9-10+'].append(date)
    
    print(f"{'SL Exits/Day':<15} {'Days':>8} {'%':>8} {'Mean PnL/Day':>15} {'Mean Intraday DD':>18}")
    print("-" * 72)
    for bucket_name, dates in sl_buckets.items():
        count = len(dates)
        pct = (count / len(test_metrics) * 100) if len(test_metrics) > 0 else 0.0
        if count > 0:
            # EXPERIMENT-012: Use date_to_metrics map
            mean_pnl = np.mean([date_to_metrics[d]['realized_pnl'] for d in dates if d in date_to_metrics])
            mean_dd = np.mean([date_to_metrics[d]['max_drawdown'] for d in dates if d in date_to_metrics])
        else:
            mean_pnl = 0.0
            mean_dd = 0.0
        print(f"{bucket_name:<15} {count:>8} {pct:>7.1f}% ${mean_pnl:>13.2f} ${mean_dd:>17.2f}")
    
    # B. Volatility quintile table
    print("\n--- Volatility Quintile Analysis (Test) ---")
    print("Days bucketed by mean ATR_norm (volatility score)")
    
    # Calculate volatility score for each day (mean ATR_norm over the day)
    day_vol_scores = []
    for date, env in episodes_with_envs:
        # EXPERIMENT-012: Use ATR_norm by name via env column mapping
        atr_norm_values = env.current_day_data[:, env._atrnorm_col]
        mean_atr_norm = np.mean(atr_norm_values)
        day_vol_scores.append((date, mean_atr_norm))
    
    # Sort by volatility and split into quintiles
    day_vol_scores_sorted = sorted(day_vol_scores, key=lambda x: x[1])
    n_days = len(day_vol_scores_sorted)
    quintile_size = n_days // 5
    
    quintiles = []
    for q in range(5):
        start_idx = q * quintile_size
        if q == 4:  # Last quintile gets remainder
            end_idx = n_days
        else:
            end_idx = (q + 1) * quintile_size
        quintile_dates = [d[0] for d in day_vol_scores_sorted[start_idx:end_idx]]
        quintiles.append((f"Q{q+1}", quintile_dates))
    
    print(f"{'Quintile':<10} {'Days':>6} {'Total PnL':>12} {'Avg PnL/Day':>13} {'PF':>6} {'Trades/Day':>12} "
          f"{'SL/Day':>8} {'Rev/Day':>8} {'Worst DD':>11}")
    print("-" * 108)
    
    for quintile_name, quintile_dates in quintiles:
        # Gather metrics for these days
        quintile_envs = [env for date, env in episodes_with_envs if date in quintile_dates]
        # EXPERIMENT-012: Use date-to-metrics map
        quintile_metrics = [date_to_metrics[d] for d in quintile_dates if d in date_to_metrics]
        
        days_count = len(quintile_dates)
        if days_count == 0:
            continue
        
        total_pnl = sum(m['realized_pnl'] for m in quintile_metrics)
        avg_pnl_per_day = total_pnl / days_count if days_count > 0 else 0.0
        
        # Calculate profit factor
        gross_profit = sum(m['realized_pnl'] for m in quintile_metrics if m['realized_pnl'] > 0)
        gross_loss = abs(sum(m['realized_pnl'] for m in quintile_metrics if m['realized_pnl'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        if pf == float('inf'):
            pf_str = "∞"
        else:
            pf_str = f"{pf:.2f}"
        
        # Trades, stop-loss, and reversal counts per day
        all_trades_in_quintile = [t for env in quintile_envs for t in env.trades_history]
        avg_trades_per_day = len(all_trades_in_quintile) / days_count if days_count > 0 else 0.0
        
        sl_exits = sum(1 for t in all_trades_in_quintile if t['exit_reason'] == 'stop_loss')
        rev_exits = sum(1 for t in all_trades_in_quintile if t['exit_reason'] == 'reversal')
        avg_sl_per_day = sl_exits / days_count if days_count > 0 else 0.0
        avg_rev_per_day = rev_exits / days_count if days_count > 0 else 0.0
        
        # Worst intraday DD in quintile
        worst_dd = min(m['max_drawdown'] for m in quintile_metrics) if quintile_metrics else 0.0
        
        print(f"{quintile_name:<10} {days_count:>6} ${total_pnl:>11.2f} ${avg_pnl_per_day:>12.2f} {pf_str:>6} "
              f"{avg_trades_per_day:>12.1f} {avg_sl_per_day:>8.1f} {avg_rev_per_day:>8.1f} ${worst_dd:>10.2f}")
    
    # C. Session slice summary
    print("\n--- Session Slice Summary (Test) --- [DEPRECATED: see EXP-014 entries-by-session below]")
    print("NOTE: This session attribution uses entry_bar which may not reflect actual execution bar.")
    print("      The EXPERIMENT-014 diagnostics section below uses trade_record['entry_session'] for")
    print("      accurate attribution. Do not use this table for session performance conclusions.")
    print("Note: Session flags from features; bar-level mapping; priority: NY > London > Asia if multiple flags.")
    
    # EXPERIMENT-012: Use proper session column mapping
    sample_env = episodes_with_envs[0][1] if episodes_with_envs else None
    if sample_env is not None:
        session_stats = {
            'Asia': {'trades': 0, 'stop_loss': 0, 'reversal': 0},
            'London': {'trades': 0, 'stop_loss': 0, 'reversal': 0},
            'NY': {'trades': 0, 'stop_loss': 0, 'reversal': 0},
            'Unknown': {'trades': 0, 'stop_loss': 0, 'reversal': 0}
        }
        
        # Scan all trades and map to sessions using entry_bar
        for date, env in episodes_with_envs:
            for trade in env.trades_history:
                entry_bar = trade['entry_bar']
                if entry_bar < len(env.current_day_data):
                    bar_data = env.current_day_data[entry_bar]
                    
                    # EXPERIMENT-012: Use session column mapping
                    session_asia = bar_data[env._session_asia_col]
                    session_london = bar_data[env._session_london_col]
                    session_ny = bar_data[env._session_ny_col]
                    
                    # Priority: NY > London > Asia (if multiple flags are set)
                    if session_ny > 0.5:
                        session = 'NY'
                    elif session_london > 0.5:
                        session = 'London'
                    elif session_asia > 0.5:
                        session = 'Asia'
                    else:
                        session = 'Unknown'
                    
                    session_stats[session]['trades'] += 1
                    if trade['exit_reason'] == 'stop_loss':
                        session_stats[session]['stop_loss'] += 1
                    if trade['exit_reason'] == 'reversal':
                        session_stats[session]['reversal'] += 1
        
        print(f"{'Session':<10} {'Trades':>10} {'Stop-Loss':>12} {'Reversal':>10}")
        print("-" * 42)
        for session, stats in session_stats.items():
            if stats['trades'] > 0:  # Only show sessions with activity
                print(f"{session:<10} {stats['trades']:>10} {stats['stop_loss']:>12} {stats['reversal']:>10}")
    else:
        print("No test episodes available for session analysis.")
    
    print("="*80)
    
    # EXPERIMENT-012 DIAGNOSTICS - ACTION DISTRIBUTION + BLOCKED ACTIONS (TEST)
    print("\n" + "="*80)
    print("EXPERIMENT-012 DIAGNOSTICS - ACTION DISTRIBUTION + BLOCKED ACTIONS")
    print("="*80)
    
    # Aggregate action counts and blocked actions from Test episodes
    print("\n--- Test 2025 Action Distribution ---")
    total_action_counts = {
        0: 0,  # HOLD
        1: 0,  # OPEN_LONG
        2: 0,  # OPEN_SHORT
        3: 0   # CLOSE_POSITION
    }
    total_masked_opens = 0
    total_blocked_reversals = 0
    total_blocked_manual_closes = 0
    total_masked_opens_cooldown_early = 0  # EXPERIMENT-013
    
    for date, env in episodes_with_envs:
        for action_id, count in env.action_counts.items():
            total_action_counts[int(action_id)] += count
        total_masked_opens += env.masked_open_count
        total_blocked_reversals += env.blocked_reversal_count
        total_blocked_manual_closes += env.blocked_manual_close_count
        total_masked_opens_cooldown_early += env.masked_open_cooldown_count  # EXPERIMENT-013
    
    total_actions = sum(total_action_counts.values())
    
    print(f"\n{'Action':<20} {'Count':>10} {'%':>8}")
    print("-" * 38)
    print(f"{'HOLD':<20} {total_action_counts[0]:>10} {(total_action_counts[0]/total_actions*100):>7.1f}%")
    print(f"{'OPEN_LONG':<20} {total_action_counts[1]:>10} {(total_action_counts[1]/total_actions*100):>7.1f}%")
    print(f"{'OPEN_SHORT':<20} {total_action_counts[2]:>10} {(total_action_counts[2]/total_actions*100):>7.1f}%")
    print(f"{'CLOSE_POSITION':<20} {total_action_counts[3]:>10} {(total_action_counts[3]/total_actions*100):>7.1f}%")
    print(f"{'TOTAL':<20} {total_actions:>10} {'100.0%':>8}")
    
    print(f"\n{'Blocked/Masked Actions':<30} {'Count':>10}")
    print("-" * 40)
    print(f"{'Masked opens (budget)':<30} {total_masked_opens:>10}")
    print(f"{'Masked opens (cooldown)':<30} {total_masked_opens_cooldown_early:>10}")  # EXPERIMENT-013
    print(f"{'Blocked reversals (min-hold)':<30} {total_blocked_reversals:>10}")
    print(f"{'Blocked manual closes (min-hold)':<30} {total_blocked_manual_closes:>10}")
    
    print("="*80)
    
    # EXPERIMENT-012 DIAGNOSTICS - ATR FALLBACK + STOP-LOSS INTEGRITY (TEST)
    print("\n" + "="*80)
    print("EXPERIMENT-012 DIAGNOSTICS - ATR FALLBACK + STOP-LOSS INTEGRITY")
    print("="*80)
    
    # Aggregate ATR fallback counts from Test
    total_atr_fallbacks = sum(env.atr_fallback_count for date, env in episodes_with_envs)
    total_bars = sum(len(env.current_day_data) for date, env in episodes_with_envs)
    
    print(f"\n--- ATR Fallback Usage (Test 2025) ---")
    print(f"Total bars processed: {total_bars:,}")
    print(f"ATR fallback applied: {total_atr_fallbacks:,} times")
    print(f"Fallback rate: {(total_atr_fallbacks/total_bars*100):.3f}%" if total_bars > 0 else "N/A")
    
    # Aggregate stop-loss integrity violations
    total_stoploss_exits = sum(sum(1 for t in env.trades_history if t['exit_reason'] == 'stop_loss') for date, env in episodes_with_envs)
    total_stoploss_positive_pnl = sum(env.stoploss_positive_pnl_count for date, env in episodes_with_envs)
    
    print(f"\n--- Stop-Loss Integrity (Test 2025) ---")
    print(f"Total stop_loss exits: {total_stoploss_exits:,}")
    print(f"Stop_loss exits with positive PnL: {total_stoploss_positive_pnl:,}")
    if total_stoploss_exits > 0:
        print(f"Integrity violation rate: {(total_stoploss_positive_pnl/total_stoploss_exits*100):.2f}%")
    
    # Aggregate exit_price_source counts
    total_exit_price_sources = {'stop_price': 0, 'bar_open': 0, 'bar_close': 0}
    for date, env in episodes_with_envs:
        for source, count in env.exit_price_source_counts.items():
            total_exit_price_sources[source] += count
    
    print(f"\n--- Exit Price Source Distribution (Test 2025) ---")
    print(f"{'Source':<15} {'Count':>10}")
    print("-" * 25)
    for source in sorted(total_exit_price_sources.keys()):
        print(f"{source:<15} {total_exit_price_sources[source]:>10}")
    
    print("="*80)
    
    # EXPERIMENT-013 DIAGNOSTICS - WEEKDAY PERFORMANCE & SESSION-FLAG SANITY
    print("\n" + "="*80)
    print("EXPERIMENT-013 DIAGNOSTICS - WEEKDAY PERFORMANCE & SESSION-FLAG SANITY")
    print("="*80)
    
    # A) Weekday performance table (Test 2025)
    print("\n--- Test 2025 Weekday Performance ---")
    from datetime import datetime as dt
    
    weekday_data = {}
    for i, date in enumerate(test_days_sorted):
        weekday = dt.strptime(date, '%Y-%m-%d').strftime('%A')
        if weekday not in weekday_data:
            weekday_data[weekday] = {
                'days': 0,
                'total_pnl': 0.0,
                'daily_pnls': [],
                'worst_day_pnl': float('inf'),
                'worst_intraday_dd': 0.0,
                'stoploss_exits': 0
            }
        
        metrics = test_metrics[i]
        weekday_data[weekday]['days'] += 1
        weekday_data[weekday]['total_pnl'] += metrics['realized_pnl']
        weekday_data[weekday]['daily_pnls'].append(metrics['realized_pnl'])
        weekday_data[weekday]['worst_day_pnl'] = min(weekday_data[weekday]['worst_day_pnl'], metrics['realized_pnl'])
        weekday_data[weekday]['worst_intraday_dd'] = min(weekday_data[weekday]['worst_intraday_dd'], metrics['max_drawdown'])
        
        # Get stop-loss exits for this day
        date_env = next((env for d, env in episodes_with_envs if d == date), None)
        if date_env:
            stoploss_count = sum(1 for t in date_env.trades_history if t['exit_reason'] == 'stop_loss')
            weekday_data[weekday]['stoploss_exits'] += stoploss_count
    
    # Print weekday table
    print(f"{'Weekday':<10} {'Days':>6} {'Total PnL':>12} {'Avg/Day':>10} {'PF':>8} {'Worst Day':>12} {'Worst DD':>12} {'Avg SL/Day':>12}")
    print("-" * 98)
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for weekday in weekday_order:
        if weekday in weekday_data:
            wd = weekday_data[weekday]
            days = wd['days']
            total_pnl = wd['total_pnl']
            avg_pnl = total_pnl / days if days > 0 else 0.0
            
            # Calculate profit factor for this weekday
            gross_profit = sum(p for p in wd['daily_pnls'] if p > 0)
            gross_loss = abs(sum(p for p in wd['daily_pnls'] if p < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
            
            worst_day = wd['worst_day_pnl']
            worst_dd = wd['worst_intraday_dd']
            avg_sl = wd['stoploss_exits'] / days if days > 0 else 0.0
            
            print(f"{weekday:<10} {days:>6} ${total_pnl:>11.2f} ${avg_pnl:>9.2f} {pf_str:>8} ${worst_day:>11.2f} ${worst_dd:>11.2f} {avg_sl:>12.2f}")
    
    # B) Session-flag sanity (Test 2025)
    print("\n--- Session-Flag Sanity Check (Test 2025 Bars) ---")
    
    total_bars_all = 0
    asia_high_count = 0
    london_high_count = 0
    ny_high_count = 0
    none_high_count = 0
    
    asia_london_overlap = 0
    london_ny_overlap = 0
    asia_ny_overlap = 0
    all3_overlap = 0
    
    for date, env in episodes_with_envs:
        day_data = env.current_day_data
        for bar_idx in range(len(day_data)):
            bar = day_data[bar_idx]
            
            asia = bar[env._session_asia_col]
            london = bar[env._session_london_col]
            ny = bar[env._session_ny_col]
            
            total_bars_all += 1
            
            asia_high = asia > 0.5
            london_high = london > 0.5
            ny_high = ny > 0.5
            
            if asia_high:
                asia_high_count += 1
            if london_high:
                london_high_count += 1
            if ny_high:
                ny_high_count += 1
            if not (asia_high or london_high or ny_high):
                none_high_count += 1
            
            # Check overlaps
            if asia_high and london_high:
                asia_london_overlap += 1
            if london_high and ny_high:
                london_ny_overlap += 1
            if asia_high and ny_high:
                asia_ny_overlap += 1
            if asia_high and london_high and ny_high:
                all3_overlap += 1
    
    print(f"Total bars: {total_bars_all:,}")
    print(f"\n{'Session Flag':<20} {'Bars > 0.5':>15} {'%':>8}")
    print("-" * 43)
    print(f"{'session_asia':<20} {asia_high_count:>15,} {(asia_high_count/total_bars_all*100):>7.1f}%")
    print(f"{'session_london':<20} {london_high_count:>15,} {(london_high_count/total_bars_all*100):>7.1f}%")
    print(f"{'session_ny':<20} {ny_high_count:>15,} {(ny_high_count/total_bars_all*100):>7.1f}%")
    print(f"{'None > 0.5':<20} {none_high_count:>15,} {(none_high_count/total_bars_all*100):>7.1f}%")
    
    print(f"\n{'Overlap':<20} {'Bars':>15} {'%':>8}")
    print("-" * 43)
    print(f"{'Asia & London':<20} {asia_london_overlap:>15,} {(asia_london_overlap/total_bars_all*100):>7.1f}%")
    print(f"{'London & NY':<20} {london_ny_overlap:>15,} {(london_ny_overlap/total_bars_all*100):>7.1f}%")
    print(f"{'Asia & NY':<20} {asia_ny_overlap:>15,} {(asia_ny_overlap/total_bars_all*100):>7.1f}%")
    print(f"{'All 3':<20} {all3_overlap:>15,} {(all3_overlap/total_bars_all*100):>7.1f}%")
    
    # C) Cooldown diagnostics (Test 2025)
    print("\n--- Stop-Loss Cooldown Gate Summary (Test 2025) ---")
    
    cooldown_triggered_days = 0
    cooldown_triggered_pnls = []
    non_cooldown_pnls = []
    cooldown_triggered_sl_exits = []
    non_cooldown_sl_exits = []
    total_masked_opens_cooldown = 0
    
    for date, env in episodes_with_envs:
        total_masked_opens_cooldown += env.masked_open_cooldown_count
        
        stoploss_count = sum(1 for t in env.trades_history if t['exit_reason'] == 'stop_loss')
        metrics = next(m for m in test_metrics if m['date'] == date)
        
        if env.cooldown_triggered:
            cooldown_triggered_days += 1
            cooldown_triggered_pnls.append(metrics['realized_pnl'])
            cooldown_triggered_sl_exits.append(stoploss_count)
        else:
            non_cooldown_pnls.append(metrics['realized_pnl'])
            non_cooldown_sl_exits.append(stoploss_count)
    
    print(f"Days where cooldown triggered: {cooldown_triggered_days} / {len(test_days_sorted)}")
    print(f"Total masked opens (cooldown): {total_masked_opens_cooldown}")
    
    if cooldown_triggered_days > 0:
        avg_pnl_triggered = np.mean(cooldown_triggered_pnls)
        avg_sl_triggered = np.mean(cooldown_triggered_sl_exits)
        print(f"Avg PnL on triggered days: ${avg_pnl_triggered:.2f}")
        print(f"Avg stop-loss exits on triggered days: {avg_sl_triggered:.2f}")
    
    if len(non_cooldown_pnls) > 0:
        avg_pnl_non_triggered = np.mean(non_cooldown_pnls)
        avg_sl_non_triggered = np.mean(non_cooldown_sl_exits)
        print(f"Avg PnL on non-triggered days: ${avg_pnl_non_triggered:.2f}")
        print(f"Avg stop-loss exits on non-triggered days: {avg_sl_non_triggered:.2f}")
    
    print("="*80)
    
    # EXPERIMENT-014 DIAGNOSTICS — SESSION PARTICIPATION & CAP TIMING
    print("\n" + "="*80)
    print("EXPERIMENT-014 DIAGNOSTICS — SESSION PARTICIPATION & CAP TIMING (TEST 2025)")
    print("="*80)
    
    # A) Entries-by-session table (Test 2025)
    # Uses trade_record['entry_session'] set at time of _open_position() — accurate and consistent.
    print("\n--- A) Entries-by-Session Table (Test 2025) ---")
    print("Source: trade_record['entry_session'] (set at open execution; accurate)")
    
    session_entry_stats = {
        'Asia':    {'entries': 0, 'pnl_sum': 0.0, 'sl_exits': 0, 'pnls': []},
        'London':  {'entries': 0, 'pnl_sum': 0.0, 'sl_exits': 0, 'pnls': []},
        'NY':      {'entries': 0, 'pnl_sum': 0.0, 'sl_exits': 0, 'pnls': []},
        'Unknown': {'entries': 0, 'pnl_sum': 0.0, 'sl_exits': 0, 'pnls': []},
    }
    
    for date, env in episodes_with_envs:
        for trade in env.trades_history:
            sess = trade.get('entry_session', 'Unknown')
            if sess not in session_entry_stats:
                sess = 'Unknown'
            session_entry_stats[sess]['entries'] += 1
            session_entry_stats[sess]['pnl_sum'] += trade['pnl']
            session_entry_stats[sess]['pnls'].append(trade['pnl'])
            if trade['exit_reason'] == 'stop_loss':
                session_entry_stats[sess]['sl_exits'] += 1
    
    total_entries_all = sum(v['entries'] for v in session_entry_stats.values())
    
    print(f"\n{'Session':<10} {'Entries':>8} {'%Entries':>10} {'PnL Sum':>12} {'Avg PnL':>10} {'SL Exits':>10}")
    print("-" * 62)
    for sess in ['Asia', 'London', 'NY', 'Unknown']:
        st = session_entry_stats[sess]
        if st['entries'] == 0:
            continue
        pct = (st['entries'] / total_entries_all * 100) if total_entries_all > 0 else 0.0
        avg_pnl = st['pnl_sum'] / st['entries'] if st['entries'] > 0 else 0.0
        print(f"{sess:<10} {st['entries']:>8} {pct:>9.1f}% ${st['pnl_sum']:>11.2f} ${avg_pnl:>9.2f} {st['sl_exits']:>10}")
    print(f"{'TOTAL':<10} {total_entries_all:>8} {'100.0%':>10}")
    
    # Also print per-episode session entry aggregates
    total_entries_in_asia_ep = sum(env.entries_in_asia for _, env in episodes_with_envs)
    total_entries_in_london_ep = sum(env.entries_in_london for _, env in episodes_with_envs)
    total_entries_in_ny_ep = sum(env.entries_in_ny for _, env in episodes_with_envs)
    print(f"\n  Cross-check from env counters (entries_in_asia/london/ny):")
    print(f"    Asia: {total_entries_in_asia_ep}, London: {total_entries_in_london_ep}, NY: {total_entries_in_ny_ep}")
    london_ny_pct = ((total_entries_in_london_ep + total_entries_in_ny_ep) / total_entries_all * 100) if total_entries_all > 0 else 0.0
    print(f"    London+NY combined: {total_entries_in_london_ep + total_entries_in_ny_ep} ({london_ny_pct:.1f}% of all entries)")
    
    # B) Cap timing table (Test 2025)
    print("\n--- B) Cap Timing: When Is the 10th Entry Made? ---")
    print("For days with >=10 entries, show bar_index of the 10th entry and which session it was in.")
    
    cap_timing_by_session = {'Asia': 0, 'London': 0, 'NY': 0, 'Unknown': 0}
    tenth_entry_bar_indices = []
    last_entry_bar_indices = []
    
    for date, env in episodes_with_envs:
        if len(env.trades_history) == 0:
            continue
        
        # Collect entry bars in chronological order
        entry_bars = sorted([t['entry_bar'] for t in env.trades_history])
        
        if len(entry_bars) >= 10:
            tenth_bar = entry_bars[9]  # 0-indexed 10th entry
            tenth_entry_bar_indices.append(tenth_bar)
            
            # Determine session of 10th entry bar
            if tenth_bar < len(env.current_day_data):
                bar_data = env.current_day_data[tenth_bar]
                ny = bar_data[env._session_ny_col]
                london = bar_data[env._session_london_col]
                asia = bar_data[env._session_asia_col]
                if ny > 0.5:
                    sess_10th = 'NY'
                elif london > 0.5:
                    sess_10th = 'London'
                elif asia > 0.5:
                    sess_10th = 'Asia'
                else:
                    sess_10th = 'Unknown'
            else:
                sess_10th = 'Unknown'
            cap_timing_by_session[sess_10th] += 1
        
        # Track last entry bar for all days
        if entry_bars:
            last_entry_bar_indices.append(entry_bars[-1])
    
    total_cap_hit = sum(cap_timing_by_session.values())
    print(f"\n  Days hitting 10-entry cap: {total_cap_hit}")
    print(f"\n{'Session of 10th Entry':<25} {'Days':>8} {'%':>8}")
    print("-" * 41)
    for sess in ['Asia', 'London', 'NY', 'Unknown']:
        count = cap_timing_by_session[sess]
        pct = (count / total_cap_hit * 100) if total_cap_hit > 0 else 0.0
        print(f"{sess:<25} {count:>8} {pct:>7.1f}%")
    
    if tenth_entry_bar_indices:
        print(f"\n  10th-entry bar index percentiles (cap-hit days):")
        print(f"    p10: {np.percentile(tenth_entry_bar_indices, 10):.0f}")
        print(f"    p50: {np.percentile(tenth_entry_bar_indices, 50):.0f}")
        print(f"    p90: {np.percentile(tenth_entry_bar_indices, 90):.0f}")
        print(f"    mean: {np.mean(tenth_entry_bar_indices):.0f}")
        print(f"    (episode window_size offset = 59; total bars/day varies)")
    
    # C) Gating effectiveness table (Test 2025)
    print("\n--- C) Gating Effectiveness: Masked OPEN Attempts by Reason ---")
    
    # Aggregate all gating counters from test episodes
    total_masked_budget = sum(env.masked_open_count for _, env in episodes_with_envs)
    total_masked_session_budget_count = sum(env.masked_open_session_budget_count for _, env in episodes_with_envs)
    total_masked_session_budget_attempts = sum(env.masked_open_session_budget_attempts for _, env in episodes_with_envs)
    total_session_budget_block_days = sum(1 for _, env in episodes_with_envs if env.session_budget_block_triggered)
    total_masked_soft_gate = sum(env.masked_open_soft_gate_attempts for _, env in episodes_with_envs)
    total_masked_hard_gate = sum(env.masked_open_hard_gate_attempts for _, env in episodes_with_envs)
    total_masked_cooldown_orig = sum(env.masked_open_cooldown_count for _, env in episodes_with_envs)
    
    total_bars_soft_gate = sum(env.bars_soft_gate_active for _, env in episodes_with_envs)
    total_bars_hard_gate = sum(env.bars_hard_gate_active for _, env in episodes_with_envs)
    total_bars_all_test = sum(len(env.current_day_data) for _, env in episodes_with_envs)
    
    print(f"\n{'Mask Reason':<35} {'Attempts':>10} {'Bars Active':>12}")
    print("-" * 57)
    print(f"{'Budget cap (>=10 trades)':<35} {total_masked_budget:>10} {'N/A':>12}")
    print(f"{'Session budget allocation':<35} {total_masked_session_budget_count:>10} {'N/A':>12}")
    print(f"{'Soft SL gate (>=2 SL, <4 SL)':<35} {total_masked_soft_gate:>10} {total_bars_soft_gate:>12,}")
    print(f"{'Hard SL gate (>=4 SL cooldown)':<35} {total_masked_hard_gate:>10} {total_bars_hard_gate:>12,}")
    
    print(f"\n  Session budget block triggered: {total_session_budget_block_days} / {len(episodes_with_envs)} days")
    print(f"  Total bars analyzed: {total_bars_all_test:,}")
    print(f"  Bars with soft gate active: {total_bars_soft_gate:,} ({(total_bars_soft_gate/total_bars_all_test*100):.2f}%)")
    print(f"  Bars with hard gate active: {total_bars_hard_gate:,} ({(total_bars_hard_gate/total_bars_all_test*100):.2f}%)")
    
    print("="*80)
    
    # EXPERIMENT-014 DIAGNOSTICS — TRAIN 2021-2024 SESSION PARTICIPATION
    print("\n" + "="*80)
    print("EXPERIMENT-014 DIAGNOSTICS — SESSION PARTICIPATION (TRAIN 2021-2024, AGGREGATE)")
    print("="*80)
    
    # Aggregate Train session counters from training episodes
    # Note: train episodes are re-run during training; we use aggregate metrics here
    # For Train, we don't have per-env access post-training, so we aggregate what we can from metrics
    # We can check if reward_components_sum captured anything related to gating
    # Instead, summarize the session budget constants and expected behavior
    print("\n  Session budget allocation constants:")
    print(f"    RESERVED_ENTRIES_LONDON = {TradingEnvironment.RESERVED_ENTRIES_LONDON}")
    print(f"    RESERVED_ENTRIES_NY = {TradingEnvironment.RESERVED_ENTRIES_NY}")
    print(f"    Asia hard limit: {TradingEnvironment.MAX_TRADES_PER_DAY - TradingEnvironment.RESERVED_ENTRIES_LONDON - TradingEnvironment.RESERVED_ENTRIES_NY} entries/day (before London/NY reached)")
    print(f"    London hard limit: {TradingEnvironment.MAX_TRADES_PER_DAY - TradingEnvironment.RESERVED_ENTRIES_NY} entries/day (before NY reached)")
    print(f"    NY limit: {TradingEnvironment.MAX_TRADES_PER_DAY} entries/day (full cap)")
    print(f"    Soft gate threshold: {TradingEnvironment.STOPLOSS_SOFT_GATE_THRESHOLD} SL exits (allow 1 entry/session)")
    print(f"    Hard gate threshold: {TradingEnvironment.STOPLOSS_COOLDOWN_THRESHOLD} SL exits (no new entries)")
    print(f"\n  Train session diagnostics require storing env objects during training.")
    print(f"  For full Train analysis, consider running a separate evaluation pass on Train dates.")
    
    print("="*80)
    
    # =========================================================================
    # EXPERIMENT-015 DIAGNOSTICS - R DISTRIBUTION & GATE INTEGRITY
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT-015 DIAGNOSTICS - R DISTRIBUTION & GATE INTEGRITY")
    print("="*80)

    # Collect all Test 2025 trades
    all_test_trades = []
    for date, env in episodes_with_envs:
        for t in env.trades_history:
            all_test_trades.append(t)

    n_trades_total = len(all_test_trades)

    # --- A) Realized R distribution table ---
    print("\n--- A) Test 2025 Realized R Distribution ---")
    bins = [
        ('<= -1R',       lambda r: r <= -1.0),
        ('(-1R, -0.2R]', lambda r: -1.0 < r <= -0.2),
        ('(-0.2R, 0.2R)',lambda r: -0.2 < r < 0.2),
        ('[0.2R, 1R)',   lambda r: 0.2 <= r < 1.0),
        ('[1R, 2R)',     lambda r: 1.0 <= r < 2.0),
        ('>= 2R',        lambda r: r >= 2.0),
        ('>= 3R',        lambda r: r >= 3.0),
    ]
    print(f"\n{'Bin':<20} {'Count':>8} {'%':>8}")
    print("-" * 36)
    for label, cond in bins:
        r_vals = [t['pnl'] / t['risk_amount'] for t in all_test_trades if t['risk_amount'] > 0]
        count = sum(1 for r in r_vals if cond(r))
        pct = (count / n_trades_total * 100) if n_trades_total > 0 else 0.0
        print(f"  {label:<18} {count:>8} {pct:>7.1f}%")
    print(f"\n  Total trades: {n_trades_total}")
    if n_trades_total > 0:
        r_vals_all = [t['pnl'] / t['risk_amount'] for t in all_test_trades if t['risk_amount'] > 0]
        print(f"  Mean R: {np.mean(r_vals_all):.3f}")
        print(f"  Median R: {np.median(r_vals_all):.3f}")

    # --- B) Session × R summary table ---
    print("\n--- B) Test 2025 Session × R Summary ---")
    sessions = ['Asia', 'London', 'NY', 'Unknown']
    print(f"\n{'Session':<12} {'Entries':>8} {'Mean R':>8} {'%>=2R':>8} {'PnL Sum':>10}")
    print("-" * 50)
    for sess in sessions:
        sess_trades = [t for t in all_test_trades if t.get('entry_session', 'Unknown') == sess]
        n_sess = len(sess_trades)
        if n_sess == 0:
            print(f"  {sess:<10} {0:>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue
        r_sess = [t['pnl'] / t['risk_amount'] for t in sess_trades if t['risk_amount'] > 0]
        mean_r = np.mean(r_sess) if r_sess else 0.0
        pct_2r = (sum(1 for r in r_sess if r >= 2.0) / len(r_sess) * 100) if r_sess else 0.0
        pnl_sum = sum(t['pnl'] for t in sess_trades)
        print(f"  {sess:<10} {n_sess:>8} {mean_r:>8.3f} {pct_2r:>7.1f}% {pnl_sum:>10.2f}")

    # --- C) Gate integrity cross-check ---
    print("\n--- C) Test 2025 Gate Integrity Cross-Check ---")
    total_bars_cooloff = sum(env.bars_cooloff_active for _, env in episodes_with_envs)
    total_masked_cooloff = sum(env.masked_open_cooloff_attempts for _, env in episodes_with_envs)
    total_masked_hard = sum(env.masked_open_hard_gate_attempts for _, env in episodes_with_envs)
    total_masked_budget = sum(env.masked_open_count for _, env in episodes_with_envs)
    total_masked_session = sum(env.masked_open_session_budget_count for _, env in episodes_with_envs)
    total_open_long = sum(env.action_counts.get(1, 0) for _, env in episodes_with_envs)
    total_open_short = sum(env.action_counts.get(2, 0) for _, env in episodes_with_envs)
    total_opens_attempted = total_open_long + total_open_short + total_masked_cooloff + total_masked_hard + total_masked_budget + total_masked_session
    total_bars_hard = sum(env.bars_hard_gate_active for _, env in episodes_with_envs)

    print(f"\n  bars_cooloff_active (EXP-015):         {total_bars_cooloff:>10,}")
    print(f"  masked_open_cooloff_attempts (EXP-015):{total_masked_cooloff:>10,}")
    print(f"  masked_open_hard_gate_attempts:        {total_masked_hard:>10,}")
    print(f"  bars_hard_gate_active:                 {total_bars_hard:>10,}")
    print(f"  masked_open_budget_attempts:           {total_masked_budget:>10,}")
    print(f"  masked_open_session_budget:            {total_masked_session:>10,}")
    print(f"  OPEN_LONG executed:                    {total_open_long:>10,}")
    print(f"  OPEN_SHORT executed:                   {total_open_short:>10,}")
    print(f"  Total OPEN attempts (exec+blocked):    {total_opens_attempted:>10,}")
    total_stoploss_positive = sum(env.stoploss_positive_pnl_count for _, env in episodes_with_envs)
    print(f"  stop_loss integrity violations:        {total_stoploss_positive:>10,}  (expected: 0)")

    print("="*80)
    
    # Calculate daily equity stats (no compounding simulation)
    train_equity_stats_daily = calculate_daily_equity_curve(train_metrics)
    # test_equity_stats already calculated earlier in print_daily_performance_summary
    
    # Save final model
    final_metadata = {
        'phase': 'final',
        'train_window': f"{train_start} to {train_end}",
        'test_window': f"{test_start} to {test_end}",
        'feature_phase': feature_phase,
        'active_features': get_active_features(feature_phase),
        'train_days': len(train_days),
        'test_days': len(test_days),
        'train_pnl': sum(m['realized_pnl'] for m in train_metrics),
        'train_avg_daily_pnl': sum(m['realized_pnl'] for m in train_metrics) / len(train_metrics) if train_metrics else 0.0,
        'train_max_intraday_dd': min(m['max_drawdown'] for m in train_metrics),
        'test_pnl': sum(m['realized_pnl'] for m in test_metrics) if test_metrics else 0.0,
        'test_avg_daily_pnl': sum(m['realized_pnl'] for m in test_metrics) / len(test_metrics) if test_metrics else 0.0,
        'test_max_intraday_dd': min(m['max_drawdown'] for m in test_metrics) if test_metrics else 0.0,
        'test_total_trades': sum(m['total_trades'] for m in test_metrics),
        'test_win_rate': np.mean([m['win_rate'] for m in test_metrics if m['total_trades'] > 0]) if test_metrics else 0.0,
        'test_daily_performance': {  # Daily P&L tracking (accurate representation)
            'daily_starting_balance': 10_000.0,
            'cumulative_pnl': test_equity_stats.get('cumulative_pnl', 0.0),
            'peak_cumulative_pnl': test_equity_stats.get('peak_cumulative_pnl', 0.0),
            'max_drawdown_from_peak': test_equity_stats.get('max_drawdown_from_peak', 0.0),
            'winning_days': test_equity_stats.get('winning_days', 0),
            'losing_days': test_equity_stats.get('losing_days', 0),
            'daily_win_rate': test_equity_stats.get('win_rate_daily', 0.0),
            'profit_factor': test_equity_stats.get('profit_factor', 0.0),
            'max_consecutive_wins': test_equity_stats.get('max_consecutive_wins', 0),
            'max_consecutive_losses': test_equity_stats.get('max_consecutive_losses', 0)
        },
        'train_daily_performance': {  # Daily P&L tracking (accurate representation)
            'daily_starting_balance': 10_000.0,
            'cumulative_pnl': train_equity_stats_daily.get('cumulative_pnl', 0.0),
            'peak_cumulative_pnl': train_equity_stats_daily.get('peak_cumulative_pnl', 0.0),
            'max_drawdown_from_peak': train_equity_stats_daily.get('max_drawdown_from_peak', 0.0),
            'winning_days': train_equity_stats_daily.get('winning_days', 0),
            'losing_days': train_equity_stats_daily.get('losing_days', 0),
            'daily_win_rate': train_equity_stats_daily.get('win_rate_daily', 0.0),
            'profit_factor': train_equity_stats_daily.get('profit_factor', 0.0)
        },
        'observed_steps_per_day': observed_steps_per_day,  # Store for reference
        'reward_function_hash': get_reward_function_hash(),
        'hyperparameters': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': epsilon_decay_steps,
            'buffer_capacity': 250_000,
            'batch_size': 64,
            'target_update_frequency': 5_000,
            'train_freq': train_freq,
            'cql_alpha': agent.cql_alpha,
            'cql_temperature': agent.cql_temperature,
            'cql_enabled': agent.cql_enabled,
            'eval_mode': eval_mode,
            'eval_temperature': eval_temperature
        },
        'agent_stats': agent.get_stats()
    }
    save_model(agent, save_dir, "final_agent", final_metadata)
    
    # Print RESULTS PACKET (EXPERIMENT-007; updated for EXPERIMENT-014)
    print_results_packet(
        experiment_id=EXPERIMENT_ID,  # Use parameterized experiment ID
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_equity_stats=train_equity_stats_daily,
        test_equity_stats=test_equity_stats,
        agent_stats=agent.get_stats(),
        epsilon_decay_steps=epsilon_decay_steps,
        eval_mode=eval_mode,
        temperature=eval_temperature,
        seeds=None,  # Seeds not currently set
        notes=""
    )
    
    # Restore logging
    restore_logging(logger)
    
    return train_metrics, test_metrics


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main training orchestration."""
    
    # Configuration
    DATA_PATH = r"C:\Users\Administrator\Desktop\RL Agent\data\US100_RL_features.parquet"
    SAVE_DIR = r"C:\Users\Administrator\Desktop\RL Agent\DQN\models"
    
    print("="*80)
    print("DQN TRADING AGENT - TRAINING ORCHESTRATION")
    print("="*80)
    print(f"Data path: {DATA_PATH}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize data loader
    data_loader = DataLoader(DATA_PATH)
    
    # Phase A: Burn-in (measures average steps per day)
    print("\n" + "="*80)
    print("STARTING PHASE A: BURN-IN WITH EPSILON CALIBRATION")
    print("="*80)
    agent, observed_steps_per_day = run_burnin_phase(
        data_loader,
        feature_phase_start=1,
        feature_phase_end=2,
        train_freq=4,
        save_dir=SAVE_DIR,
        checkpoint_every=100
    )
    
    # Phase B: Walk-forward validation (uses calibrated steps/day)
    print("\n" + "="*80)
    print("STARTING PHASE B: WALK-FORWARD VALIDATION")
    print("="*80)
    run_walkforward_validation(
        data_loader,
        observed_steps_per_day=observed_steps_per_day,
        feature_phase=2,
        train_freq=4,
        save_dir=SAVE_DIR,
        checkpoint_every=100
    )
    
    # Phase C: Final train & test (uses calibrated steps/day)
    print("\n" + "="*80)
    print("STARTING PHASE C: FINAL TRAIN & TEST")
    print("="*80)
    run_final_train_test(
        data_loader,
        observed_steps_per_day=observed_steps_per_day,
        feature_phase=4,
        train_freq=4,
        save_dir=SAVE_DIR,
        checkpoint_every=100
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"All models saved to: {SAVE_DIR}")
    print(f"Epsilon calibration used: {observed_steps_per_day:.1f} steps/day")


if __name__ == "__main__":
    main()