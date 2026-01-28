from .base_game import BaseGame
import torch


class FuzzyObjectGame(BaseGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**8,
        context_size=(2, 3),
        confusion_prob=0.0,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__(
            game_instances=game_instances,
            agents=agents,
            objects=objects,
            memory=memory,
            context_size=context_size,
            vocab_size=vocab_size,
            device=device,
        )
        self.fig_prefix = "fuzzy_object_game"
        if not (0.0 <= float(confusion_prob) <= 1.0):
            raise ValueError("confusion_prob must be in [0, 1].")
        self.confusion_prob = torch.as_tensor(confusion_prob, device=device, dtype=torch.float32)

        # obsturct perception channel

    def perception_channel(self, flat_counts: torch.Tensor) -> torch.Tensor:
        """Remove best match with probability obstruct_prob. Optimized to avoid CPU syncs."""
        G = flat_counts.size(0)
        
        # obstruction mask for all instances
        obstruct_mask = torch.rand(G, device=self.device) < self.confusion_prob  # (G,)
        
        # reshape to (G, objects, memory)
        reshaped = flat_counts.view(G, -1, self.memory)
        
        # Find best object per instance (max count across all memory slots)
        # Get the flat index of the maximum, then extract object index
        best_flat_idx = reshaped.view(G, -1).argmax(dim=-1)  # (G,)
        best_object_idx = best_flat_idx // self.memory  # (G,)
        
        # Create mask to zero out best object's memory slots
        # one_hot: (G, objects)
        obj_mask = torch.nn.functional.one_hot(best_object_idx, num_classes=reshaped.size(1)).bool()  # (G, objects)
        
        # Expand to cover memory: (G, objects, memory)
        zero_mask = obj_mask.unsqueeze(-1).expand(-1, -1, self.memory)
        
        # Zero out best object where obstruct_mask is True
        zeroed = torch.where(zero_mask, torch.zeros_like(reshaped), reshaped)
        
        # Apply obstruction conditionally using torch.where
        result = torch.where(
            obstruct_mask.unsqueeze(-1).unsqueeze(-1),  # (G, 1, 1)
            zeroed,
            reshaped
        )
        
        return result.view(G, -1)

