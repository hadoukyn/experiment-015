"""
DQN Agent for Trading Environment
Implements Double DQN with Dueling Architecture and 1D CNN Encoder.
Supports action masking and is fully compatible with trading_env.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, Dict


class DuelingQRDQNetwork(nn.Module):
    """
    Dueling QR-DQN (Quantile Regression DQN) with 1D CNN encoder.
    
    Architecture:
    - Input: (batch_size, 60, 29) - 60 timesteps, 29 features
    - Encoder: 1D Convolutions over time dimension (same as before)
    - Head: Dueling split into Value and Advantage streams (now outputting quantiles)
    - Output: Quantile values of shape (batch_size, n_actions, n_quantiles)
    
    EXPERIMENT-008: Implements distributional RL with quantile regression
    """
    
    def __init__(self, state_shape: Tuple[int, int], n_actions: int, n_quantiles: int = 51):
        """
        Initialize the network.
        
        Args:
            state_shape: Shape of state (window_size, n_features) = (60, 29)
            n_actions: Number of discrete actions = 4
            n_quantiles: Number of quantiles to estimate (default: 51)
        """
        super(DuelingQRDQNetwork, self).__init__()
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        
        window_size, n_features = state_shape
        
        # 1D CNN Encoder over time dimension (UNCHANGED from DuelingDQNetwork)
        # Input: (batch_size, n_features, window_size) after transpose
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Calculate flattened size after convolutions
        # Since we use padding='same' equivalent, size remains window_size
        self.flatten_size = 128 * window_size
        
        # Shared feature layer (UNCHANGED)
        self.fc_shared = nn.Linear(self.flatten_size, 512)
        
        # Dueling streams - now output quantiles instead of scalars
        # Value stream: estimates V(s, τ) for each quantile τ
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_quantiles)  # Output: (batch, n_quantiles)
        )
        
        # Advantage stream: estimates A(s, a, τ) for each action and quantile
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions * n_quantiles)  # Output: (batch, n_actions * n_quantiles)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor of shape (batch_size, 60, 29)
            
        Returns:
            Quantile values Z(s,a,τ) of shape (batch_size, n_actions, n_quantiles)
        """
        batch_size = x.size(0)
        
        # Transpose for 1D convolution: (batch, features, time)
        x = x.transpose(1, 2)  # (batch_size, 29, 60)
        
        # Convolutional encoder (UNCHANGED)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Shared layer
        x = F.relu(self.fc_shared(x))
        
        # Dueling streams - now per quantile
        value = self.value_stream(x)  # (batch_size, n_quantiles)
        advantage = self.advantage_stream(x)  # (batch_size, n_actions * n_quantiles)
        
        # Reshape advantage to (batch_size, n_actions, n_quantiles)
        advantage = advantage.view(batch_size, self.n_actions, self.n_quantiles)
        
        # Expand value for broadcasting: (batch_size, 1, n_quantiles)
        value = value.unsqueeze(1)
        
        # Combine using dueling formula per quantile:
        # Z(s,a,τ) = V(s,τ) + (A(s,a,τ) - mean_a A(s,a,τ))
        quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return quantiles  # (batch_size, n_actions, n_quantiles)


class ReplayBuffer:
    """
    Uniform random sampling replay buffer.
    Stores transitions: (state, action, reward, next_state, done, current_action_mask, next_action_mask)
    
    Note: current_action_mask is needed for proper CQL regularization to exclude invalid actions
    from logsumexp computation.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, current_action_mask: np.ndarray,
             next_action_mask: np.ndarray) -> None:
        """
        Store a transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            current_action_mask: Valid actions mask for current state
            next_action_mask: Valid actions mask for next_state
        """
        self.buffer.append((state, action, reward, next_state, done, 
                          current_action_mask, next_action_mask))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones, 
                            current_action_masks, next_action_masks)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, current_action_masks, next_action_masks = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        current_action_masks = torch.FloatTensor(np.array(current_action_masks))
        next_action_masks = torch.FloatTensor(np.array(next_action_masks))
        
        return states, actions, rewards, next_states, dones, current_action_masks, next_action_masks
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Double QR-DQN Agent with Dueling Architecture.
    
    Features (EXPERIMENT-008):
    - QR-DQN (Quantile Regression DQN) for distributional RL
    - Risk-averse action selection using lower-quantile mean
    - Double DQN for stable learning
    - Dueling architecture for better value estimation
    - 1D CNN encoder for temporal patterns
    - Action masking support
    - Epsilon-greedy exploration
    - Quantile Huber loss for robustness
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int],
        n_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 300_000,
        buffer_capacity: int = 250_000,
        batch_size: int = 64,
        target_update_frequency: int = 5_000,
        cql_alpha: float = 0.2,
        cql_temperature: float = 1.0,
        cql_enabled: bool = True,
        n_quantiles: int = 51,
        quantile_huber_kappa: float = 1.0,
        risk_fraction: float = 0.25,
        device: Optional[str] = None
    ):
        """
        Initialize QR-DQN agent (EXPERIMENT-008).
        
        Args:
            state_shape: Shape of state (window_size, n_features)
            n_actions: Number of discrete actions
            learning_rate: Learning rate for Adam optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon linearly
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_frequency: Steps between target network updates
            cql_alpha: CQL regularization coefficient (default 0.2)
            cql_temperature: Temperature for CQL logsumexp (default 1.0)
            cql_enabled: Whether to enable CQL regularization (default True)
            n_quantiles: Number of quantiles for distributional RL (default 51)
            quantile_huber_kappa: Huber loss threshold for quantile regression (default 1.0)
            risk_fraction: Fraction of lower quantiles for risk-averse action selection (default 0.25)
            device: Device to use ('cuda' or 'cpu')
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # CQL parameters
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature
        self.cql_enabled = cql_enabled
        
        # QR-DQN parameters (EXPERIMENT-008)
        self.n_quantiles = n_quantiles
        self.quantile_huber_kappa = quantile_huber_kappa
        self.risk_fraction = risk_fraction
        
        # Calculate number of quantiles for risk-averse statistic
        self.k_risk_quantiles = max(1, int(n_quantiles * risk_fraction))
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks (EXPERIMENT-008: QR-DQN networks)
        self.policy_net = DuelingQRDQNetwork(state_shape, n_actions, n_quantiles).to(self.device)
        self.target_net = DuelingQRDQNetwork(state_shape, n_actions, n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        
        # EXPERIMENT-006: Separate env_steps (for epsilon decay) from update_steps (for optimizer updates)
        # env_steps = total environment interactions (every action selected during training)
        # update_steps = total optimizer updates (only when train_step() is called)
        self.env_steps = 0      # Environment interaction steps (for epsilon decay)
        self.update_steps = 0   # Optimizer update steps (for target network updates)
        self.training_steps = 0  # DEPRECATED: kept for backward compatibility, use update_steps instead
        
    def select_action(
        self, 
        state: np.ndarray, 
        action_mask: np.ndarray, 
        training: bool = True,
        eval_mode: str = 'argmax',
        temperature: float = 0.15
    ) -> int:
        """
        Select action using epsilon-greedy policy with action masking.
        
        EXPERIMENT-008: Uses risk-averse statistic (lower-quantile mean) for exploitation/evaluation
        
        During training: Uses epsilon-greedy exploration with risk-averse exploitation
        During evaluation: Uses either argmax (risk-averse) or softmax sampling
        
        Args:
            state: Current state (window_size, n_features)
            action_mask: Binary mask of valid actions (n_actions,)
            training: Whether in training mode (affects exploration)
            eval_mode: Evaluation policy mode ('argmax' or 'softmax'). Only used when training=False.
            temperature: Temperature for softmax sampling. Only used when eval_mode='softmax'.
            
        Returns:
            Selected action index
        """
        # Update epsilon if training
        if training:
            # EXPERIMENT-006: Increment environment steps counter (used for epsilon decay)
            # This happens every time an action is selected during training
            self.env_steps += 1
            self.epsilon = self._get_epsilon()
        
        # Epsilon-greedy exploration (training only)
        if training and random.random() < self.epsilon:
            # Random action from valid actions
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        
        # Get quantile values: (1, n_actions, n_quantiles)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self.policy_net(state_tensor)  # (1, n_actions, n_quantiles)
            quantiles = quantiles.cpu().numpy()[0]  # (n_actions, n_quantiles)
            
            # Compute risk-averse statistic: mean of lowest k quantiles per action
            # Sort quantiles for each action and take mean of bottom k
            risk_statistics = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                sorted_quantiles = np.sort(quantiles[a])  # Sort in ascending order
                # Take mean of lowest k quantiles
                risk_statistics[a] = sorted_quantiles[:self.k_risk_quantiles].mean()
            
            # Apply action mask
            masked_risk_stats = np.where(action_mask == 1, risk_statistics, -np.inf)
            
            # Training mode: argmax selection based on risk statistic
            if training:
                action = np.argmax(masked_risk_stats)
            # Evaluation mode: use specified eval_mode
            else:
                if eval_mode == 'argmax':
                    # Deterministic: select action with highest risk statistic
                    action = np.argmax(masked_risk_stats)
                elif eval_mode == 'softmax':
                    # Stochastic: softmax sampling with temperature based on risk statistic
                    # Apply temperature scaling
                    scaled_stats = masked_risk_stats / temperature
                    # Compute softmax probabilities
                    exp_stats = np.exp(scaled_stats - np.max(scaled_stats))  # Subtract max for numerical stability
                    probs = exp_stats / np.sum(exp_stats)
                    # Sample action according to probabilities
                    action = np.random.choice(self.n_actions, p=probs)
                else:
                    raise ValueError(f"Unknown eval_mode: {eval_mode}. Must be 'argmax' or 'softmax'.")
            
            return int(action)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        current_action_mask: np.ndarray,
        next_action_mask: np.ndarray
    ) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            current_action_mask: Valid actions mask for current state
            next_action_mask: Valid actions mask for next_state
        """
        self.replay_buffer.push(state, action, reward, next_state, done, 
                              current_action_mask, next_action_mask)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step with Quantile Regression (QR-DQN).
        
        EXPERIMENT-008: Uses quantile Huber loss instead of TD loss.
        CQL is disabled for this experiment.
        
        Returns:
            Dictionary with loss components if training occurred:
                - 'total_loss': Quantile regression loss
                - 'td_loss': 0.0 (for compatibility)
                - 'cql_loss': 0.0 (disabled for EXPERIMENT-008)
            None if buffer too small
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, current_action_masks, next_action_masks = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        current_action_masks = current_action_masks.to(self.device)
        next_action_masks = next_action_masks.to(self.device)
        
        # ============ QR-DQN Quantile Regression Loss ============
        # Get current quantiles for taken actions: (batch_size, n_quantiles)
        current_quantiles = self.policy_net(states)  # (batch_size, n_actions, n_quantiles)
        current_quantiles = current_quantiles.gather(
            1, 
            actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles)
        ).squeeze(1)  # (batch_size, n_quantiles)
        
        # Compute target quantiles using Double DQN with action masking
        with torch.no_grad():
            # Get quantiles from policy network for next states
            next_quantiles_policy = self.policy_net(next_states)  # (batch_size, n_actions, n_quantiles)
            
            # Compute risk-averse statistics for action selection
            # Sort quantiles and take mean of lowest k for each action
            risk_statistics = torch.zeros(self.batch_size, self.n_actions, device=self.device)
            for a in range(self.n_actions):
                sorted_quantiles, _ = torch.sort(next_quantiles_policy[:, a, :], dim=1)
                risk_statistics[:, a] = sorted_quantiles[:, :self.k_risk_quantiles].mean(dim=1)
            
            # Mask invalid actions by setting their risk statistics to -inf
            masked_risk_stats = torch.where(
                next_action_masks == 1,
                risk_statistics,
                torch.tensor(-float('inf'), device=self.device)
            )
            
            # Use policy network to select best valid actions based on risk statistic
            next_actions = masked_risk_stats.argmax(1)  # (batch_size,)
            
            # Use target network to evaluate those actions
            next_quantiles_target = self.target_net(next_states)  # (batch_size, n_actions, n_quantiles)
            next_quantiles = next_quantiles_target.gather(
                1,
                next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles)
            ).squeeze(1)  # (batch_size, n_quantiles)
            
            # Compute target quantiles: reward + gamma * next_quantiles * (1 - done)
            # Broadcast reward and done to quantile dimension
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles * (1 - dones).unsqueeze(1)
            # target_quantiles: (batch_size, n_quantiles)
        
        # Compute quantile Huber loss (Dabney et al. 2018)
        # Expand dimensions for pairwise TD errors
        current_quantiles_expanded = current_quantiles.unsqueeze(2)  # (batch, n_quantiles, 1)
        target_quantiles_expanded = target_quantiles.unsqueeze(1)  # (batch, 1, n_quantiles)
        
        # TD errors: θ_i - θ'_j for all pairs i, j
        td_errors = target_quantiles_expanded - current_quantiles_expanded  # (batch, n_quantiles, n_quantiles)
        
        # Quantile midpoints (τ)
        tau = torch.linspace(0.0, 1.0, self.n_quantiles + 1, device=self.device)
        tau_hat = (tau[:-1] + tau[1:]) / 2.0  # Midpoints: (n_quantiles,)
        
        # Huber loss component
        huber_loss = torch.where(
            td_errors.abs() <= self.quantile_huber_kappa,
            0.5 * td_errors ** 2,
            self.quantile_huber_kappa * (td_errors.abs() - 0.5 * self.quantile_huber_kappa)
        )  # (batch, n_quantiles, n_quantiles)
        
        # Quantile regression weight: |τ - I{TD error < 0}|
        quantile_weight = torch.abs(
            tau_hat.view(1, -1, 1) - (td_errors < 0).float()
        )  # (batch, n_quantiles, n_quantiles)
        
        # Quantile regression loss
        quantile_loss = (quantile_weight * huber_loss).mean()
        
        # Total loss (CQL disabled for EXPERIMENT-008)
        total_loss = quantile_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # EXPERIMENT-006: Increment update steps counter (optimizer updates)
        # This is separate from env_steps which tracks environment interactions
        self.update_steps += 1
        self.training_steps += 1  # Keep for backward compatibility
        
        # Update target network if needed
        if self.update_steps % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Return loss components for logging (CQL always 0 for EXPERIMENT-008)
        return {
            'total_loss': total_loss.item(),
            'td_loss': 0.0,  # Not applicable for QR-DQN, kept for compatibility
            'cql_loss': 0.0  # CQL disabled for EXPERIMENT-008
        }
    
    def update_target_network(self) -> None:
        """
        Hard update: copy policy network weights to target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _get_epsilon(self) -> float:
        """
        Compute current epsilon using three-phase decay schedule.
        
        EXPERIMENT-006: Now uses env_steps (environment interactions) instead of
        training_steps (optimizer updates) for epsilon decay calculation.
        
        Three-Phase Schedule:
        - Phase 1 (0-40% of decay_steps): Linear decay from 1.0 to 0.1
        - Phase 2 (40-70% of decay_steps): Linear decay from 0.1 to 0.02
        - Phase 3 (70-100% of decay_steps): Fixed at 0.01
        
        This schedule ensures:
        1. Aggressive exploration early (Phase 1)
        2. Transition to exploitation (Phase 2)
        3. Pure exploitation for convergence (Phase 3)
        
        Returns:
            Current epsilon value
        """
        # Phase boundaries (as fractions of total decay steps)
        phase1_end = 0.40  # 40% of decay steps
        phase2_end = 0.70  # 70% of decay steps
        
        # Calculate progress through training using env_steps
        progress = self.env_steps / self.epsilon_decay_steps
        
        # Phase 1: Linear decay from 1.0 to 0.1
        if progress < phase1_end:
            # Map [0, 0.40] -> [1.0, 0.1]
            phase_progress = progress / phase1_end
            epsilon = 1.0 - (0.9 * phase_progress)
            return epsilon
        
        # Phase 2: Linear decay from 0.1 to 0.02
        elif progress < phase2_end:
            # Map [0.40, 0.70] -> [0.1, 0.02]
            phase_progress = (progress - phase1_end) / (phase2_end - phase1_end)
            epsilon = 0.1 - (0.08 * phase_progress)
            return epsilon
        
        # Phase 3: Fixed at 0.01
        else:
            return 0.01
    
    def save(self, path: str) -> None:
        """
        Save agent state to file.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'env_steps': self.env_steps,           # EXPERIMENT-006: Save env_steps
            'update_steps': self.update_steps,     # EXPERIMENT-006: Save update_steps
            'training_steps': self.training_steps, # Backward compatibility
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """
        Load agent state from file.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # EXPERIMENT-006: Load env_steps and update_steps with backward compatibility
        self.env_steps = checkpoint.get('env_steps', 0)
        self.update_steps = checkpoint.get('update_steps', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        
        self.epsilon = checkpoint['epsilon']
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current agent statistics.
        
        Returns:
            Dictionary with agent metrics
        """
        return {
            'env_steps': self.env_steps,           # EXPERIMENT-006: Environment interaction steps
            'update_steps': self.update_steps,     # EXPERIMENT-006: Optimizer update steps
            'training_steps': self.training_steps, # Backward compatibility (same as update_steps)
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
        }