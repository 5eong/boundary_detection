"""
Lookahead Optimizer wrapper.

PyTorch implementation of the Lookahead optimizer wrapper that can be applied
to any base optimizer to improve convergence and reduce variance.

Reference: "Lookahead Optimizer: k steps forward, 1 step back"
           by Zhang et al. (https://arxiv.org/abs/1907.08610)
"""

import math
import torch
import itertools as it
from torch.optim import Optimizer
from collections import defaultdict
from typing import Dict, Any, Optional, Union


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    
    This optimizer wraps around any base optimizer and implements the Lookahead
    algorithm, which maintains two sets of weights: fast weights (updated by the
    base optimizer) and slow weights (updated by the Lookahead algorithm).
    
    The algorithm periodically updates the slow weights by interpolating between
    the current slow weights and the fast weights, then resets the fast weights
    to the slow weights.
    
    Args:
        optimizer: Base optimizer to wrap
        alpha: Linear interpolation factor (0.0 to 1.0)
        k: Number of fast weight updates before slow weight update
        pullback_momentum: How to handle momentum during interpolation
                          ("none", "pullback", "reset")
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        alpha: float = 0.5,
        k: int = 6,
        pullback_momentum: str = "none"
    ):
        """
        Initialize Lookahead optimizer.
        
        Args:
            optimizer: Inner/base optimizer
            alpha: Linear interpolation factor for slow weights update
            k: Number of lookahead steps before updating slow weights
            pullback_momentum: Momentum handling strategy
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not k >= 1:
            raise ValueError(f'Invalid lookahead steps: {k}')
        
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        
        if pullback_momentum not in ["reset", "pullback", "none"]:
            raise ValueError(f'Invalid pullback_momentum: {pullback_momentum}')
        self.pullback_momentum = pullback_momentum
        
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters as slow weights
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                
                # Initialize cached momentum if using pullback
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state for deserialization."""
        self.__dict__.update(state)

    def zero_grad(self) -> None:
        """Clear gradients of all parameters."""
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary of the base optimizer."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into the base optimizer."""
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self) -> None:
        """
        Backup current parameters and load cached slow weights.
        
        This is useful for performing evaluation on the slow weights,
        which typically generalize better than the fast weights.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self) -> None:
        """
        Clear backup and restore fast weights.
        
        This restores the fast weights after evaluation on slow weights.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        # Perform base optimizer step
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        # Check if it's time for lookahead update
        if self.step_counter >= self.k:
            self.step_counter = 0
            
            # Perform lookahead update on slow weights
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    
                    # Update slow weights: θ_slow = θ_slow + α(θ_fast - θ_slow)
                    # Equivalent to: θ_slow = (1-α)θ_slow + αθ_fast
                    p.data.mul_(self.alpha).add_(
                        param_state['cached_params'], alpha=1.0 - self.alpha
                    )
                    
                    # Cache updated slow weights
                    param_state['cached_params'].copy_(p.data)
                    
                    # Handle momentum based on strategy
                    if self.pullback_momentum == "pullback":
                        # Pull back momentum buffer
                        if p in self.optimizer.state and "momentum_buffer" in self.optimizer.state[p]:
                            internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                            self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(
                                self.alpha
                            ).add_(param_state["cached_mom"], alpha=1.0 - self.alpha)
                            param_state["cached_mom"].copy_(
                                self.optimizer.state[p]["momentum_buffer"]
                            )
                    
                    elif self.pullback_momentum == "reset":
                        # Reset momentum buffer
                        if p in self.optimizer.state and "momentum_buffer" in self.optimizer.state[p]:
                            self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer."""
        self.optimizer.add_param_group(param_group)
        
        # Initialize Lookahead state for new parameters
        for p in param_group['params']:
            param_state = self.state[p]
            param_state['cached_params'] = torch.zeros_like(p.data)
            param_state['cached_params'].copy_(p.data)
            
            if self.pullback_momentum == "pullback":
                param_state['cached_mom'] = torch.zeros_like(p.data)

    @property
    def defaults(self) -> Dict[str, Any]:
        """Get default parameters."""
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: Dict[str, Any]) -> None:
        """Set default parameters."""
        self.optimizer.defaults = defaults

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        format_string = self.__class__.__name__ + ' ('
        format_string += f'alpha={self.alpha}, '
        format_string += f'k={self.k}, '
        format_string += f'pullback_momentum={self.pullback_momentum}, '
        format_string += f'base_optimizer={self.optimizer.__class__.__name__}'
        format_string += ')'
        return format_string


def create_lookahead_optimizer(
    base_optimizer_class: type,
    parameters,
    lookahead_alpha: float = 0.5,
    lookahead_k: int = 6,
    pullback_momentum: str = "none",
    **base_optimizer_kwargs
) -> Lookahead:
    """
    Factory function to create a Lookahead optimizer with a base optimizer.
    
    Args:
        base_optimizer_class: Class of the base optimizer (e.g., torch.optim.Adam)
        parameters: Model parameters to optimize
        lookahead_alpha: Lookahead alpha parameter
        lookahead_k: Lookahead k parameter
        pullback_momentum: Momentum handling strategy
        **base_optimizer_kwargs: Arguments for the base optimizer
        
    Returns:
        Lookahead optimizer wrapping the base optimizer
        
    Example:
        >>> import torch.optim as optim
        >>> model = MyModel()
        >>> optimizer = create_lookahead_optimizer(
        ...     optim.Adam,
        ...     model.parameters(),
        ...     lookahead_alpha=0.6,
        ...     lookahead_k=10,
        ...     lr=0.001,
        ...     weight_decay=1e-4
        ... )
    """
    base_optimizer = base_optimizer_class(parameters, **base_optimizer_kwargs)
    return Lookahead(
        base_optimizer,
        alpha=lookahead_alpha,
        k=lookahead_k,
        pullback_momentum=pullback_momentum
    )
