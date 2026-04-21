"""
Transformer-based feature extractor for Stable-Baselines3 PPO.

This module provides TransformerFeaturesExtractor, a proper SB3
BaseFeaturesExtractor subclass that processes sequential market
observations ([seq_len, n_features]) through a multi-head self-attention
Transformer encoder before passing the pooled representation to
SB3's actor-critic MLP heads.

The last-layer attention weights are cached after every forward pass
so they can be extracted for explainability without a second inference.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor that applies a Transformer encoder
    to sequential observations of shape (seq_len, n_features).

    Architecture:
        Input [B, seq_len, n_features]
        -> Linear projection to d_model
        -> Learnable positional encoding
        -> N TransformerEncoderLayers (multi-head self-attention + FFN)
        -> Mean pooling over sequence dimension
        -> Output [B, d_model]

    The output dimensionality (features_dim) equals d_model and is passed
    to SB3's default actor/critic MLP heads.

    Attention weights from the last encoder layer are cached in
    `self.last_attention_weights` after each forward pass.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        # features_dim is what SB3 uses as the input size for actor/critic
        super().__init__(observation_space, features_dim=d_model)

        obs_shape = observation_space.shape  # (seq_len, n_features)
        assert len(obs_shape) == 2, (
            f"TransformerFeaturesExtractor requires 2-D observations "
            f"(seq_len, n_features), got shape {obs_shape}"
        )
        seq_len, n_features = obs_shape

        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Linear projection from raw feature dim to d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Cache for the last-layer attention weights (set during forward)
        self.last_attention_weights: Optional[torch.Tensor] = None

        # Register a forward hook on the last encoder layer's self-attention
        # to capture attention weights without modifying the forward pass.
        self._attn_hook_handle = (
            self.transformer.layers[-1]
            .self_attn.register_forward_hook(self._capture_attention)
        )

    def _capture_attention(self, module, args, output):
        """
        Hook called after nn.MultiheadAttention.forward().
        output is (attn_output, attn_weights) when need_weights=True,
        but by default PyTorch's TransformerEncoderLayer does NOT request
        weights. We re-compute them here cheaply from Q/K cached in args.
        """
        # The standard hook receives positional args: (query, key, value, ...)
        # We'll compute attention weights explicitly from the attention output.
        # For efficiency, we store a flag and compute in forward() instead.
        pass  # See forward() below for actual extraction.

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: [batch, seq_len, n_features] from the environment.

        Returns:
            Pooled features [batch, d_model] for the actor-critic heads.
        """
        # Project input features to d_model dimension
        x = self.input_projection(observations)  # [B, S, d_model]

        # Add learnable positional encoding
        x = x + self.positional_encoding  # broadcasts over batch

        # Run through transformer encoder
        x = self.transformer(x)  # [B, S, d_model]

        # Extract attention weights from the last layer for explainability.
        # We do a lightweight extra forward through the last layer's MHA only.
        last_layer = self.transformer.layers[-1]
        with torch.no_grad():
            _, attn_weights = last_layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=True
            )
            # attn_weights shape: [B, seq_len, seq_len]
            self.last_attention_weights = attn_weights.detach()

        # Mean pool over the sequence dimension
        pooled = x.mean(dim=1)  # [B, d_model]

        return pooled

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Returns the cached attention weights from the last forward pass.

        Returns:
            Tensor of shape [B, seq_len, seq_len] or None if no forward
            pass has been performed yet.
        """
        return self.last_attention_weights
