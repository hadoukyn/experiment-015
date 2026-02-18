"""
RL Data Loader & Window Manager for Intraday Trading

This module provides infrastructure for loading and preprocessing time-series data
for reinforcement learning trading systems. It handles:
- Efficient parquet data loading
- Trading hours filtering
- Episode construction (one day = one episode)
- Rolling 60-bar state windows
- Walk-forward train/validation/test splits
- Data validation and logging

Author: Senior Python Engineer
Phase: 1 - Data Infrastructure
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from datetime import datetime, time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLDataLoader:
    """
    Data loader for RL trading system with rolling window state construction.
    
    Responsibilities:
    - Load and validate parquet data
    - Filter to trading hours
    - Construct daily episodes
    - Provide rolling 60-bar states
    - Support walk-forward splits
    
    Attributes:
        data: Full filtered DataFrame
        episodes: Dict mapping episode_id to (start_idx, end_idx, date)
        window_size: Size of rolling state window (default 60)
        feature_cols: List of 29 feature column names
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 60,
        validate: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to US100_RL_features.parquet
            window_size: Number of bars in rolling window
            validate: Whether to run validation checks
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        
        logger.info(f"Initializing RLDataLoader from {self.data_path}")
        logger.info(f"Window size: {window_size}")
        
        # Load and preprocess data
        self.data = self._load_data()
        self.feature_cols = self._identify_feature_columns()
        
        # Filter to trading hours
        self.data = self._filter_trading_hours(self.data)
        
        # Construct episodes
        self.episodes = self._construct_episodes()
        
        # Validation
        if validate:
            self._validate_data()
            self._log_statistics()
    
    def _load_data(self) -> pd.DataFrame:
        """Load parquet file and perform initial checks."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info("Loading parquet file...")
        df = pd.read_parquet(self.data_path)
        
        # Ensure Timestamp column exists and is datetime
        if 'Timestamp' not in df.columns:
            raise ValueError("Timestamp column not found in data")
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        
        return df
    
    def _identify_feature_columns(self) -> List[str]:
        """
        Identify the 29 feature columns (excluding OHLC and Timestamp).
        
        Returns:
            List of feature column names
        """
        excluded = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        feature_cols = [col for col in self.data.columns if col not in excluded]
        
        if len(feature_cols) != 29:
            logger.warning(
                f"Expected 29 feature columns, found {len(feature_cols)}: {feature_cols}"
            )
        
        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")
        return feature_cols
    
    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to allowed trading hours (broker time).
        
        Rules:
        - Monday-Thursday: 01:05 → 23:50
        - Friday: 01:05 → 22:55
        - Saturday & Sunday: excluded
        
        Args:
            df: Input DataFrame with Timestamp column
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering to trading hours...")
        
        initial_rows = len(df)
        
        # Extract time components
        df = df.copy()
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6
        df['Time'] = df['Timestamp'].dt.time
        
        # Define trading hours
        monday_thursday_start = time(1, 5)
        monday_thursday_end = time(23, 50)
        friday_start = time(1, 5)
        friday_end = time(22, 55)
        
        # Create filter mask
        mask = pd.Series(False, index=df.index)
        
        # Monday-Thursday (0-3)
        weekday_mask = df['DayOfWeek'].isin([0, 1, 2, 3])
        time_mask = (df['Time'] >= monday_thursday_start) & (df['Time'] <= monday_thursday_end)
        mask |= (weekday_mask & time_mask)
        
        # Friday (4)
        friday_mask = df['DayOfWeek'] == 4
        friday_time_mask = (df['Time'] >= friday_start) & (df['Time'] <= friday_end)
        mask |= (friday_mask & friday_time_mask)
        
        # Apply filter
        df = df[mask].copy()
        
        # Drop helper columns
        df = df.drop(columns=['DayOfWeek', 'Time'])
        df = df.reset_index(drop=True)
        
        filtered_rows = len(df)
        removed_rows = initial_rows - filtered_rows
        
        logger.info(f"Filtered {removed_rows:,} rows outside trading hours")
        logger.info(f"Remaining rows: {filtered_rows:,}")
        
        return df
    
    def _construct_episodes(self) -> Dict[int, Tuple[int, int, str]]:
        """
        Construct daily episodes from filtered data.
        
        Episode rules:
        - One episode = one trading day
        - Episodes must be sequential
        - Each episode must contain ≥ 60 valid bars
        - Episode boundaries reset rolling windows
        
        Returns:
            Dict mapping episode_id to (start_idx, end_idx, date_str)
        """
        logger.info("Constructing episodes...")
        
        # Extract date from timestamp
        self.data['Date'] = self.data['Timestamp'].dt.date
        
        # Group by date
        grouped = self.data.groupby('Date')
        
        episodes = {}
        episode_id = 0
        
        for date, group in grouped:
            start_idx = group.index[0]
            end_idx = group.index[-1]
            num_bars = len(group)
            
            # Only include episodes with sufficient bars
            if num_bars >= self.window_size:
                episodes[episode_id] = (start_idx, end_idx, str(date))
                episode_id += 1
            else:
                logger.debug(f"Skipping {date}: only {num_bars} bars (need {self.window_size})")
        
        logger.info(f"Constructed {len(episodes)} valid episodes")
        
        return episodes
    
    def _validate_data(self):
        """Run validation checks on loaded data."""
        logger.info("Running validation checks...")
        
        # Check for NaNs
        nan_counts = self.data[self.feature_cols].isna().sum()
        if nan_counts.any():
            logger.error("NaN values found in features:")
            logger.error(nan_counts[nan_counts > 0])
            raise ValueError("NaN values detected in feature columns")
        
        # Check for infinities
        inf_mask = np.isinf(self.data[self.feature_cols].values)
        if inf_mask.any():
            inf_counts = inf_mask.sum(axis=0)
            logger.error("Infinite values found in features:")
            for i, count in enumerate(inf_counts):
                if count > 0:
                    logger.error(f"  {self.feature_cols[i]}: {count} inf values")
            raise ValueError("Infinite values detected in feature columns")
        
        # Check chronological ordering
        timestamps = self.data['Timestamp'].values
        if not np.all(timestamps[:-1] <= timestamps[1:]):
            logger.error("Data is not chronologically ordered")
            raise ValueError("Timestamps are not in chronological order")
        
        # Check episode continuity
        for ep_id, (start_idx, end_idx, date) in self.episodes.items():
            episode_dates = self.data.loc[start_idx:end_idx, 'Date'].unique()
            if len(episode_dates) > 1:
                logger.error(f"Episode {ep_id} spans multiple dates: {episode_dates}")
                raise ValueError(f"Episode {ep_id} contains multiple dates")
        
        logger.info("✓ All validation checks passed")
    
    def _log_statistics(self):
        """Log dataset statistics."""
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        
        # Overall stats
        logger.info(f"Total rows: {len(self.data):,}")
        logger.info(f"Total episodes: {len(self.episodes):,}")
        logger.info(f"Feature columns: {len(self.feature_cols)}")
        
        # Date coverage
        min_date = self.data['Timestamp'].min()
        max_date = self.data['Timestamp'].max()
        logger.info(f"Date range: {min_date} to {max_date}")
        
        # Episode statistics
        episode_lengths = []
        for start_idx, end_idx, date in self.episodes.values():
            episode_lengths.append(end_idx - start_idx + 1)
        
        episode_lengths = np.array(episode_lengths)
        logger.info(f"Bars per episode:")
        logger.info(f"  Min: {episode_lengths.min()}")
        logger.info(f"  Max: {episode_lengths.max()}")
        logger.info(f"  Mean: {episode_lengths.mean():.1f}")
        logger.info(f"  Median: {np.median(episode_lengths):.1f}")
        
        # Check for state-producing capacity
        usable_bars_per_episode = episode_lengths - (self.window_size - 1)
        total_usable_bars = usable_bars_per_episode.sum()
        logger.info(f"Total state-producing bars: {total_usable_bars:,}")
        logger.info(f"  (First {self.window_size - 1} bars of each episode cannot produce states)")
        
        logger.info("=" * 60)
    
    def get_state(self, row_index: int) -> Optional[np.ndarray]:
        """
        Get rolling window state for a given row index.
        
        The state consists of the previous 60 bars (including current bar).
        Returns None if insufficient history exists.
        
        Args:
            row_index: Index in self.data
            
        Returns:
            np.ndarray of shape (60, 29) or None if insufficient history
        """
        # Find which episode this row belongs to
        episode_id = None
        for ep_id, (start_idx, end_idx, date) in self.episodes.items():
            if start_idx <= row_index <= end_idx:
                episode_id = ep_id
                episode_start_idx = start_idx
                break
        
        if episode_id is None:
            return None
        
        # Check if we have enough history within this episode
        bars_since_episode_start = row_index - episode_start_idx + 1
        
        if bars_since_episode_start < self.window_size:
            return None
        
        # Extract window
        window_start = row_index - self.window_size + 1
        window_end = row_index + 1
        
        state = self.data.loc[window_start:row_index, self.feature_cols].values
        
        assert state.shape == (self.window_size, len(self.feature_cols)), \
            f"Expected shape ({self.window_size}, {len(self.feature_cols)}), got {state.shape}"
        
        return state
    
    def iterate_episodes(
        self,
        split: str = "train",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Iterator[Tuple[int, int, int, str]]:
        """
        Iterate over episodes for a given split or date range.
        
        Args:
            split: "train", "validation", or "test" (uses default date ranges)
            start_date: Optional custom start date (YYYY-MM-DD)
            end_date: Optional custom end date (YYYY-MM-DD)
            
        Yields:
            Tuple of (episode_id, start_idx, end_idx, date_str)
        """
        # Get episode IDs for the split
        episode_ids = self.get_episode_indices(
            split=split,
            start_date=start_date,
            end_date=end_date
        )
        
        for ep_id in episode_ids:
            start_idx, end_idx, date = self.episodes[ep_id]
            yield ep_id, start_idx, end_idx, date
    
    def get_episode_indices(
        self,
        split: str = "train",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[int]:
        """
        Get episode IDs for a given split or custom date range.
        
        Default date ranges:
        - train: 2021-01-04 to 2022-12-31
        - validation: rolling 3-month windows
        - test: 2025-01-01 to 2025-12-31
        
        Args:
            split: "train", "validation", or "test"
            start_date: Optional custom start date (YYYY-MM-DD)
            end_date: Optional custom end date (YYYY-MM-DD)
            
        Returns:
            List of episode IDs
        """
        # Use custom dates if provided
        if start_date is not None and end_date is not None:
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()
        else:
            # Default date ranges
            if split == "train":
                start = pd.to_datetime("2021-01-04").date()
                end = pd.to_datetime("2022-12-31").date()
            elif split == "validation":
                # This is a placeholder - in practice, validation windows
                # would be dynamically calculated for walk-forward
                start = pd.to_datetime("2023-01-01").date()
                end = pd.to_datetime("2023-03-31").date()
            elif split == "test":
                start = pd.to_datetime("2025-01-01").date()
                end = pd.to_datetime("2025-12-31").date()
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")
        
        # Filter episodes by date range
        episode_ids = []
        for ep_id, (start_idx, end_idx, date_str) in self.episodes.items():
            date = pd.to_datetime(date_str).date()
            if start <= date <= end:
                episode_ids.append(ep_id)
        
        return episode_ids
    
    def get_episode_metadata(self, episode_id: int) -> Dict:
        """
        Get metadata for a specific episode.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Dict with episode metadata
        """
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not found")
        
        start_idx, end_idx, date = self.episodes[episode_id]
        num_bars = end_idx - start_idx + 1
        num_usable_bars = num_bars - (self.window_size - 1)
        
        # Get first and last timestamps
        first_timestamp = self.data.loc[start_idx, 'Timestamp']
        last_timestamp = self.data.loc[end_idx, 'Timestamp']
        
        return {
            'episode_id': episode_id,
            'date': date,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_bars': num_bars,
            'num_usable_bars': num_usable_bars,
            'first_timestamp': first_timestamp,
            'last_timestamp': last_timestamp,
            'first_usable_idx': start_idx + self.window_size - 1
        }
    
    def get_walk_forward_splits(
        self,
        initial_train_start: str = "2021-01-04",
        initial_train_end: str = "2022-12-31",
        validation_months: int = 3,
        final_test_start: str = "2025-01-01",
        final_test_end: str = "2025-12-31"
    ) -> List[Dict]:
        """
        Generate expanding walk-forward splits for training.
        
        Args:
            initial_train_start: Start of initial training period
            initial_train_end: End of initial training period
            validation_months: Length of validation window in months
            final_test_start: Start of final test period
            final_test_end: End of final test period
            
        Returns:
            List of split configurations, each containing:
            - train_start, train_end
            - val_start, val_end
            - fold_id
        """
        splits = []
        
        # Parse dates
        train_start = pd.to_datetime(initial_train_start)
        train_end = pd.to_datetime(initial_train_end)
        test_start = pd.to_datetime(final_test_start)
        
        fold_id = 0
        
        while True:
            # Validation window starts after current training end
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1)
            
            # Stop if validation window reaches test period
            if val_end >= test_start:
                break
            
            # Check if we have data for this validation window
            val_episodes = self.get_episode_indices(
                split="validation",
                start_date=val_start.strftime("%Y-%m-%d"),
                end_date=val_end.strftime("%Y-%m-%d")
            )
            
            if len(val_episodes) > 0:
                splits.append({
                    'fold_id': fold_id,
                    'train_start': train_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'val_start': val_start.strftime("%Y-%m-%d"),
                    'val_end': val_end.strftime("%Y-%m-%d")
                })
                fold_id += 1
            
            # Expand training window to include validation period
            train_end = val_end
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        
        return splits
    
    def get_features_array(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get feature array for a range of indices.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            
        Returns:
            np.ndarray of shape (n_bars, 29)
        """
        return self.data.loc[start_idx:end_idx, self.feature_cols].values
    
    def get_ohlc(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get OHLC data for a range of indices.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            
        Returns:
            np.ndarray of shape (n_bars, 4) with columns [Open, High, Low, Close]
        """
        return self.data.loc[start_idx:end_idx, ['Open', 'High', 'Low', 'Close']].values
    
    def get_timestamps(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get timestamps for a range of indices.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)
            
        Returns:
            np.ndarray of timestamps
        """
        return self.data.loc[start_idx:end_idx, 'Timestamp'].values


def main():
    """Example usage and testing."""
    # Example data path (adjust as needed)
    data_path = "C:\\Users\\Administrator\\Desktop\\RL Agent\\data\\US100_RL_features.parquet"
    
    # Initialize loader
    loader = RLDataLoader(data_path, window_size=60)
    
    # Get train episodes
    train_episodes = loader.get_episode_indices(split="train")
    print(f"\nTrain episodes: {len(train_episodes)}")
    
    # Example: iterate through first 3 train episodes
    print("\nFirst 3 training episodes:")
    for i, (ep_id, start_idx, end_idx, date) in enumerate(
        loader.iterate_episodes(split="train")
    ):
        if i >= 3:
            break
        
        metadata = loader.get_episode_metadata(ep_id)
        print(f"\nEpisode {ep_id} ({date}):")
        print(f"  Total bars: {metadata['num_bars']}")
        print(f"  Usable bars: {metadata['num_usable_bars']}")
        
        # Get a state from the middle of the episode
        mid_idx = (start_idx + end_idx) // 2
        state = loader.get_state(mid_idx)
        if state is not None:
            print(f"  State shape: {state.shape}")
            print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
    
    # Generate walk-forward splits
    print("\n" + "=" * 60)
    print("WALK-FORWARD SPLITS")
    print("=" * 60)
    splits = loader.get_walk_forward_splits()
    for split in splits[:3]:  # Show first 3 folds
        print(f"\nFold {split['fold_id']}:")
        print(f"  Train: {split['train_start']} to {split['train_end']}")
        print(f"  Val:   {split['val_start']} to {split['val_end']}")


if __name__ == "__main__":
    main()