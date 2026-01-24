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
        prune_step=50,
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
        if not (0.0 <= confusion_prob <= 1.0):
            raise ValueError("confusion_prob must be in [0, 1].")
        self.confusion_prob = confusion_prob

        # obsturct perception channel

    def perception_channel(self, flat_counts: torch.Tensor) -> torch.Tensor:
        """Remove best match with probability obstruct_prob."""
        if self.confusion_prob <= 0.0:
            return flat_counts
        
        obstruct_mask = torch.rand(flat_counts.size(0), device=self.device) < self.confusion_prob
        if obstruct_mask.any():
            masked_counts = flat_counts[obstruct_mask]
            # reshape to (n_masked, objects, memory) and find best object
            reshaped = masked_counts.view(masked_counts.size(0), -1, self.memory)
            best_object_idx = reshaped.sum(dim=-1).argmax(dim=-1)
            # zero out all memory slots for best object
            flat_counts = flat_counts.clone()
            reshaped_clone = reshaped.clone()
            reshaped_clone[torch.arange(reshaped.size(0), device=self.device), best_object_idx, :] = 0.0
            flat_counts[obstruct_mask] = reshaped_clone.view(masked_counts.size(0), -1)
        
        return flat_counts

