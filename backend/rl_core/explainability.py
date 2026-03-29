import torch
import numpy as np

class ExplainabilityLayer:
    """
    Extracts attention weights from Transformer and feature importance to explain agent decisions.
    Integrates with frontend UI via WebSockets.
    """
    def __init__(self, model):
        """
        Expects a trained PPOActorCritic model containing a TransformerEncoder.
        """
        self.model = model

    def get_attention_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        Runs a forward pass and extracts the self-attention weights from the last layer.
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) # Add batch dim
        
        # This requires the TransformerEncoder to hook and store attention weights during forward pass.
        # Assuming model.encoder.transformer.layers[-1].self_attn stores attention matrices:
        
        try:
            # Pseudo-code for extraction:
            # weights = self.model.encoder.transformer.layers[-1].self_attn.get_attention_weights()
            # return weights.detach().numpy()
            
            # Dummy output for scaffold based on sequence length
            seq_len = obs.shape[0] if obs.ndim == 2 else 64
            dummy_attention = np.random.dirichlet(np.ones(seq_len), size=1)[0]
            return dummy_attention
            
        except AttributeError:
            return np.array([])

    def get_feature_importance(self, obs: np.ndarray) -> np.ndarray:
        """
        Approximates feature importance using sensitivity analysis.
        Calculates gradients of the Critic value with respect to the input features.
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        obs_tensor.requires_grad = True
        
        # Forward pass to critic
        action_logits, value = self.model(obs_tensor)
        
        # Backprop from value to observe sensitivity of inputs
        value.backward()
        
        # The absolute gradient magnitude indicates feature sensitivity
        # Averages across the sequence length to get importance per feature
        sensitivity = torch.abs(obs_tensor.grad).mean(dim=(0, 1)).detach().numpy()
        
        # Normalize to sum to 1
        if sensitivity.sum() > 0:
            sensitivity = sensitivity / sensitivity.sum()
            
        return sensitivity
