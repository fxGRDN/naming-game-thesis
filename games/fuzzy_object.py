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

    def set_words_for_hearers(self, hearers, contexts, words, topics) -> None:
        """
        Hearer can:
            1. know heard word and see named object in context -> reinforce association
            2. know heard word but not see named object in context -> choose object from context
            3. not know heard word -> choose object from context and associate with word

        """

        self.successful_communications = torch.zeros_like(
            self.successful_communications
        )

        objects_from_context = self.choose_object_from_context(contexts)

        # 1. check if memory holds the word of objects in context
        hearer_words = self.state[self.instance_ids, hearers, :, :, 0]

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
                self.state[self.instance_ids, hearers, objects_from_context, :, 1] == 0
            )  # (n_missing_hearers,)
            set_word_mask = (~found_per_hearer) & avaible_memory_slot.any(dim=-1)

            if set_word_mask.any():
                first_empty_idx = avaible_memory_slot.float().argmax(dim=-1)

                self.state[
                    self.instance_ids[set_word_mask],
                    hearers[set_word_mask],
                    objects_from_context[set_word_mask],
                    first_empty_idx[set_word_mask],
                ] = torch.stack(
                    [words[set_word_mask], torch.ones_like(words[set_word_mask])],
                    dim=-1,
                )

        # 1. find memory idx and reinforce
        if found_per_hearer.any():

            # Get counts for matching positions, zero elsewhere
            hearer_counts = self.state[
                self.instance_ids, hearers, :, :, 1
            ]  # (n_hearers, objects, memory)
            match_counts = hearer_counts * matches  # (n_hearers, objects, memory)

            # Flatten (objects, memory) -> find best (object, memory) combo per hearer
            flat_counts = match_counts.view(
                match_counts.size(0), -1
            )  # (n_hearers, objects*memory)
            best_flat_idx = flat_counts.argmax(dim=-1)  # (n_hearers,)

            # Convert flat index back to (object_idx, memory_idx)
            best_object_idx = best_flat_idx // self.memory
            best_memory_idx = best_flat_idx % self.memory

            if self.confusion_prob > 0.0:
                found_idx = torch.nonzero(found_per_hearer, as_tuple=False).squeeze(1)
                n_found = found_idx.numel()
                if n_found > 0:
                    confuse_mask = (
                        torch.rand(n_found, device=self.device) < self.confusion_prob
                    )
                    if confuse_mask.any():
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
                            best_object_idx[found_idx] = subset_obj
                            best_memory_idx[found_idx] = subset_mem

            self.successful_communications = (topics == best_object_idx).to(
                torch.float32
            )

            # Damp everything by -1 (clamped at 0)
            self.state[
                self.instance_ids[found_per_hearer],
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                :,
                1,
            ] = torch.clamp(
                self.state[
                    self.instance_ids[found_per_hearer],
                    hearers[found_per_hearer],
                    best_object_idx[found_per_hearer],
                    :,
                    1,
                ]
                - 1,
                min=0,
            )

            # Reinforce: increment the best match
            self.state[
                self.instance_ids[found_per_hearer],
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                best_memory_idx[found_per_hearer],
                1,
            ] = torch.clamp(
                self.state[
                    self.instance_ids[found_per_hearer],
                    hearers[found_per_hearer],
                    best_object_idx[found_per_hearer],
                    best_memory_idx[found_per_hearer],
                    1,
                ]
                + 2,
                max=100,
            )