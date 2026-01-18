from .base_game import BaseGame
import torch


class FuzzyObjectGame(BaseGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**15,
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
            prune_step=prune_step,
            device=device,
        )
        self.fig_prefix = "fuzzy_object_game"
        if not (0.0 <= confusion_prob <= 1.0):
            raise ValueError("confusion_prob must be in [0, 1].")
        self.confusion_prob = confusion_prob

    def perception_channel(
        self, found_per_hearer, hearers, contexts, best_object_idx, best_memory_idx
    ):
        """
        Apply perception noise to the found object/memory indices.
        With probability confusion_prob, replace the found object with a random one from context.
        """
        if self.confusion_prob <= 0.0:
            return best_object_idx, best_memory_idx

        found_idx = torch.nonzero(found_per_hearer, as_tuple=False).squeeze(1)
        n_found = found_idx.numel()
        if n_found == 0:
            return best_object_idx, best_memory_idx

        confuse_mask = torch.rand(n_found, device=self.device) < self.confusion_prob
        if not confuse_mask.any():
            return best_object_idx, best_memory_idx

        # Work on the subset of found hearers
        subset_obj = best_object_idx[found_idx].clone()
        subset_mem = best_memory_idx[found_idx].clone()

        subset_contexts = contexts[found_idx][confuse_mask]
        random_objects = self.choose_object_from_context(subset_contexts)

        available_slots = (
            self.state[
                self.instance_ids[found_idx[confuse_mask]],
                hearers[found_idx[confuse_mask]],
                random_objects,
                :,
                1,
            ]
            == 0
        )
        has_slot = available_slots.any(dim=-1)
        if has_slot.any():
            first_empty = available_slots.float().argmax(dim=-1)

            confused_idx = confuse_mask.nonzero(as_tuple=False).squeeze(1)
            target_idx = confused_idx[has_slot]

            subset_obj[target_idx] = random_objects[has_slot]
            subset_mem[target_idx] = first_empty[has_slot]

            # write back to full tensors
            best_object_idx = best_object_idx.clone()
            best_memory_idx = best_memory_idx.clone()
            best_object_idx[found_idx] = subset_obj
            best_memory_idx[found_idx] = subset_mem

        
        


        return best_object_idx, best_memory_idx