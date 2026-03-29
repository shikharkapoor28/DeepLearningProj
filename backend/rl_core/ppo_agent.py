import torch
import torch.nn as nn
from typing import Tuple

class TransformerEncoder(nn.Module):
    """
    Extracts temporal features from sequential market data.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # PyTorch native transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        embedded = self.embedding(x)
        features = self.transformer(embedded)
        
        # Average pooling over the sequence or selecting the CLS token / last element
        # Here we average pool the sequential features
        pooled_features = features.mean(dim=1) 
        
        return pooled_features


class PPOActorCritic(nn.Module):
    """
    PPO Policy Network integrating a Transformer Encoder.
    """
    def __init__(self, encoder: nn.Module, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        
        # Shared extractor dim (from encoder d_model)
        # Assuming d_model from encoder matches input to actor/critic
        d_model = 128 
        
        # Actor Head: Outputs parameters for Action Distribution
        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic Head: Outputs Value Estimate
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class PPOTrainer:
    """
    Skeleton for PPO Training Loop.
    """
    def __init__(self, env, config: dict):
        self.env = env
        self.config = config
        
        # Initialization logic
        input_dim = config.get("n_features", 10)
        action_dim = env.action_space.shape[0]
        
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        self.model = PPOActorCritic(self.encoder, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        
    def train_step(self):
        """
        Executes real data collection in the env and updates the model.
        """
        # 1. Rollout trajectory (states, actions, rewards, log_probs, values)
        # 2. Compute Advantages (GAE)
        # 3. Optimize surrogate loss over epochs
        pass
