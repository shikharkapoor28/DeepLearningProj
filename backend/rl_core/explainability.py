"""
Explainability utilities for the Transformer-based PPO trading agent.

Provides two methods:
1. **Attention weight extraction**: Reads the cached attention weights from
   TransformerFeaturesExtractor after a forward pass — no random/dummy data.
2. **Gradient-based feature importance**: Computes input sensitivity via
   backpropagation through the SB3 policy network.

Both methods operate on SB3's PPO model object directly.
"""

import torch
import numpy as np
from typing import Optional, List
from stable_baselines3 import PPO

from rl_core.ppo_agent import TransformerFeaturesExtractor


class ExplainabilityLayer:
    """
    Extracts real attention weights and gradient-based feature importance
    from a trained PPO model that uses TransformerFeaturesExtractor.
    """

    FEATURE_NAMES: List[str] = [
        "Return", "Volatility", "RSI_14", "MACD", "Turbulence", "Volume"
    ]

    def __init__(self, model: PPO):
        """
        Args:
            model: A Stable-Baselines3 PPO model whose policy uses
                   TransformerFeaturesExtractor.
        """
        self.model = model
        self._extractor = self._get_extractor()

    def _get_extractor(self) -> Optional[TransformerFeaturesExtractor]:
        """
        Navigates SB3's policy object tree to find the
        TransformerFeaturesExtractor instance.
        """
        try:
            extractor = self.model.policy.features_extractor
            if isinstance(extractor, TransformerFeaturesExtractor):
                return extractor
        except AttributeError:
            pass
        return None

    def get_attention_weights(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs a forward pass through the features extractor and returns the
        real self-attention weights from the last Transformer layer.

        Args:
            obs: Observation array of shape (seq_len, n_features).

        Returns:
            Attention weight matrix of shape (seq_len, seq_len) averaged
            over heads, or None if the extractor is not available.
        """
        if self._extractor is None:
            return None

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)  # add batch dim

        with torch.no_grad():
            self._extractor(obs_tensor)

        attn = self._extractor.get_attention_weights()
        if attn is None:
            return None

        # attn shape: [B, seq_len, seq_len] — take first batch element
        return attn[0].cpu().numpy()

    def get_feature_importance(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes gradient-based feature importance by backpropagating
        through the policy's value head.

        The absolute gradient magnitude w.r.t. each input feature,
        averaged over the sequence dimension, indicates how sensitive
        the critic's value estimate is to that feature.

        Args:
            obs: Observation array of shape (seq_len, n_features).

        Returns:
            Normalized importance scores of shape (n_features,) summing to 1.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)

        obs_tensor = obs_tensor.clone().detach().requires_grad_(True)

        # SB3 policy: features_extractor -> mlp_extractor -> value_net
        features = self.model.policy.features_extractor(obs_tensor)
        latent_pi, latent_vf = self.model.policy.mlp_extractor(features)
        value = self.model.policy.value_net(latent_vf)

        value.backward()

        if obs_tensor.grad is None:
            n_feat = obs.shape[-1] if obs.ndim >= 2 else 6
            return np.ones(n_feat) / n_feat

        # Absolute gradient magnitude averaged over batch and sequence
        sensitivity = torch.abs(obs_tensor.grad).mean(dim=(0, 1)).detach().cpu().numpy()

        total = sensitivity.sum()
        if total > 0:
            sensitivity = sensitivity / total

        return sensitivity

    def get_top_features(self, obs: np.ndarray, top_k: int = 4) -> List[str]:
        """
        Returns the names of the top-k most important features for this
        observation based on gradient sensitivity.
        """
        importance = self.get_feature_importance(obs)
        indices = np.argsort(importance)[::-1][:top_k]
        names = self.FEATURE_NAMES
        return [names[i] if i < len(names) else f"Feature_{i}" for i in indices]
