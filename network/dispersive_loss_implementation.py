#!/usr/bin/env python3
"""
Dispersive Loss Implementation for DPPO-MLP
==========================================

Core dispersive loss implementation for MLP-based diffusion models.
Provides representation regularization during pretraining phase.
"""

import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class DispersiveLoss:
    """Dispersive Loss implementation for representation regularization."""
    
    @staticmethod
    def compute_dispersive_loss(
        representations: torch.Tensor, 
        loss_type: str = "infonce_l2", 
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        Compute dispersive loss for representation regularization.
        
        Args:
            representations: (batch_size, feature_dim) intermediate representations
            loss_type: "infonce_l2", "infonce_cosine", "hinge", "covariance"
            temperature: Temperature parameter for scaling distances
        
        Returns:
            dispersive_loss: Scalar tensor
        """
        if representations.size(0) <= 1:
            return torch.tensor(0.0, device=representations.device)
        
        if loss_type == "infonce_l2":
            distances = torch.cdist(representations, representations, p=2)
            mask = ~torch.eye(distances.size(0), dtype=bool, device=distances.device)
            distances = distances[mask]
            exp_neg_dist = torch.exp(-distances / temperature)
            dispersive_loss = torch.log(torch.mean(exp_neg_dist)+1e-8)
            
        elif loss_type == "infonce_cosine":
            representations_norm = F.normalize(representations, p=2, dim=1)
            similarities = torch.mm(representations_norm, representations_norm.t())
            distances = 1 - similarities
            mask = ~torch.eye(distances.size(0), dtype=bool, device=distances.device)
            distances = distances[mask]
            exp_neg_dist = torch.exp(-distances / temperature)
            dispersive_loss = torch.log(torch.mean(exp_neg_dist)+1e-8)
            
        elif loss_type == "hinge":
            distances = torch.cdist(representations, representations, p=2)
            mask = ~torch.eye(distances.size(0), dtype=bool, device=distances.device)
            distances = distances[mask]
            epsilon = 1.0
            hinge_losses = torch.clamp(epsilon - distances, min=0) ** 2
            dispersive_loss = torch.mean(hinge_losses)
            
        elif loss_type == "covariance":
            representations_centered = representations - representations.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(representations_centered.t(), representations_centered) / (representations.size(0) - 1)
            mask = ~torch.eye(cov_matrix.size(0), dtype=bool, device=cov_matrix.device)
            off_diagonal = cov_matrix[mask]
            dispersive_loss = torch.sum(off_diagonal ** 2)
            
        else:
            raise ValueError(f"Unknown dispersive loss type: {loss_type}")
            
        return dispersive_loss


class DispersiveLossIntegration:
    """Integration class for adding dispersive loss to diffusion models."""
    
    def __init__(
        self,
        use_dispersive_loss: bool = False,
        dispersive_loss_weight: float = 0.2,
        dispersive_loss_temperature: float = 0.5,
        dispersive_loss_type: str = "infonce_l2",
        dispersive_loss_layer: str = "early"
    ):
        self.use_dispersive_loss = use_dispersive_loss
        self.dispersive_loss_weight = dispersive_loss_weight
        self.dispersive_loss_temperature = dispersive_loss_temperature
        self.dispersive_loss_type = dispersive_loss_type
        self.dispersive_loss_layer = dispersive_loss_layer
        self._log_dispersive_loss = 0.0
    
    def get_intermediate_representations(
        self, 
        model: torch.nn.Module,
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Get intermediate representations and compute dispersive loss for MLP models.
        
        Args:
            model: The MLP network model to hook
            x: Input tensor
            t: Timestep tensor
            cond: Conditioning tensor
            
        Returns:
            dispersive_loss: Computed dispersive loss tensor
        """
        representations = []
        hooks = []
        
        def hook_fn(module, input, output):
            if len(output.shape) > 2:
                output_flat = output.view(output.size(0), -1)
            else:
                output_flat = output
            representations.append(output_flat)
        
        try:
            # Hook MLP layers
            mlp_layers = [layer for layer in model.layers if hasattr(layer, 'weight')]
            
            if len(mlp_layers) > 0:
                if self.dispersive_loss_layer == "early":
                    hook = mlp_layers[len(mlp_layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "mid":
                    hook = mlp_layers[len(mlp_layers) // 2].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "late":
                    hook = mlp_layers[3 * len(mlp_layers) // 4].register_forward_hook(hook_fn)
                    hooks.append(hook)
                elif self.dispersive_loss_layer == "all":
                    for layer in mlp_layers:
                        hook = layer.register_forward_hook(hook_fn)
                        hooks.append(hook)
            
            # Forward pass to capture representations
            _ = model(x, t, cond=cond)
            
            # Compute dispersive loss
            if representations:
                if self.dispersive_loss_layer == "all":
                    total_loss = 0
                    for repr_tensor in representations:
                        total_loss += DispersiveLoss.compute_dispersive_loss(
                            repr_tensor, 
                            self.dispersive_loss_type, 
                            self.dispersive_loss_temperature
                        )
                    dispersive_loss = total_loss / len(representations)
                else:
                    dispersive_loss = DispersiveLoss.compute_dispersive_loss(
                        representations[0], 
                        self.dispersive_loss_type, 
                        self.dispersive_loss_temperature
                    )
            else:
                dispersive_loss = torch.tensor(0.0, device=x.device)
                
        finally:
            for hook in hooks:
                hook.remove()
        
        return dispersive_loss


# Configuration template for dispersive loss
def get_dispersive_config():
    """Get dispersive loss configuration template."""
    return {
        "use_dispersive_loss": True,
        "dispersive_loss_weight": 0.2, 
        "dispersive_loss_temperature": 0.5,
        "dispersive_loss_type": "infonce_l2",
        "dispersive_loss_layer": "early"  # Options: "early", "mid", "late", "all"
    }