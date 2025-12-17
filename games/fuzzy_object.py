from .base_game import BaseGame
import torch


class FuzzyObjectGame(BaseGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=10**6,
        prune_step=100,
        confusion_prob=0.0,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__(
            game_instances=game_instances,
            agents=agents,
            objects=objects,
            memory=memory,
            vocab_size=vocab_size,
            prune_step=prune_step,
            device=device,
        )
        self.fig_prefix = "fuzzy_object_game"
        if not (0.0 <= confusion_prob <= 1.0):
            raise ValueError("confusion_prob must be in [0, 1].")
        self.confusion_prob = confusion_prob

    def set_words_for_hearers(self, hearers, contexts, words) -> None:
        """
        Hearer can:
            1. know heard word and see named object in context -> reinforce association
               (with confusion_prob chance, reinforce a random object from context instead)
            2. know heard word but not see named object in context -> choose object from context
            3. not know heard word -> choose object from context and associate with word

        """

        self.successful_communications = 0

        objects_from_context = self.choose_object_from_context(contexts)

        # 1. check if memory holds the word of objects in context
        hearer_words = self.state[hearers, :, :, 0]

        # mask words not in context
        hearer_words_in_context = hearer_words * (contexts > 0).unsqueeze(-1)

        # words: (n_hearers,)
        # hearer_words_in_context: (n_hearers, objects, memory)

        matches = (
            hearer_words_in_context == words[:, None, None]
        )  # (n_hearers, objects, memory) bool
        found_per_hearer = matches.any(dim=(1, 2))  # (n_hearers,) bool

        # 2. and 3.
        if (~found_per_hearer).any():

            avaible_memory_slot = (
                self.state[hearers, objects_from_context, :, 1] == 0
            )  # (n_missing_hearers,)
            set_word_mask = (~found_per_hearer) & avaible_memory_slot.any(dim=-1)

            if set_word_mask.any():
                first_empty_idx = avaible_memory_slot.float().argmax(dim=-1)

                self.state[
                    hearers[set_word_mask],
                    objects_from_context[set_word_mask],
                    first_empty_idx[set_word_mask],
                ] = torch.stack(
                    [words[set_word_mask], torch.ones_like(words[set_word_mask])],
                    dim=-1,
                ).long()

        # 1. find memory idx and reinforce
        if found_per_hearer.any():

            self.successful_communications = found_per_hearer.sum().item()
            # two objects could have the same word, so we take one with highest association

            # Get counts for matching positions, zero elsewhere
            hearer_counts = self.state[hearers, :, :, 1]  # (n_hearers, objects, memory)
            match_counts = hearer_counts * matches  # (n_hearers, objects, memory)

            # Flatten (objects, memory) -> find best (object, memory) combo per hearer
            flat_counts = match_counts.view(
                match_counts.size(0), -1
            )  # (n_hearers, objects*memory)
            best_flat_idx = flat_counts.argmax(dim=-1)  # (n_hearers,)

            # Convert flat index back to (object_idx, memory_idx)
            best_object_idx = best_flat_idx // self.memory
            best_memory_idx = best_flat_idx % self.memory

            # With confusion_prob, pick a random object from context instead
            if self.confusion_prob > 0.0:
                n_found = found_per_hearer.sum().item()
                confuse_mask = (
                    torch.rand(n_found, device=self.device) < self.confusion_prob
                )

                if confuse_mask.any():
                    # Get random objects from context for confused hearers
                    random_objects = self.choose_object_from_context(
                        contexts[found_per_hearer][confuse_mask]
                    )

                    # Find first available memory slot for the random object
                    confused_hearers = hearers[found_per_hearer][confuse_mask]
                    available_slots = (
                        self.state[confused_hearers, random_objects, :, 1] == 0
                    )
                    has_slot = available_slots.any(dim=-1)
                    first_empty = available_slots.float().argmax(dim=-1)

                    # Update best_object_idx and best_memory_idx for confused hearers
                    # Map back to full found_per_hearer indices
                    found_indices = torch.arange(n_found, device=self.device)
                    confused_indices = found_indices[confuse_mask]

                    best_object_idx = best_object_idx.clone()
                    best_memory_idx = best_memory_idx.clone()

                    # For those with available slot, use random object
                    best_object_idx[found_per_hearer][confused_indices[has_slot]] = (
                        random_objects[has_slot]
                    )
                    best_memory_idx[found_per_hearer][confused_indices[has_slot]] = (
                        first_empty[has_slot]
                    )

            # Reinforce: increment the count
            self.state[
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                best_memory_idx[found_per_hearer],
                1,
            ] += 1
