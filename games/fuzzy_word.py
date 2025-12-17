import torch
from .base_game import BaseGame


class FuzzyWordGame(BaseGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**8,
        prune_step=100,
        flip_prob=0.0,
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
        self.fig_prefix = "fuzzy_word_game"

        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1].")
        self.flip_prob = flip_prob
        # Number of bits needed to represent vocab_size
        self.word_bits = (vocab_size - 1).bit_length()

    def flip_random_bit(self, words: torch.Tensor) -> torch.Tensor:
        """
        Flip a single random bit in each word.
        """
        # Random bit position per word
        bit_pos = torch.randint(
            0, self.word_bits, (words.shape[0],), device=self.device
        )
        flip_mask = 1 << bit_pos
        return words ^ flip_mask

    def get_words_from_speakers(self, speakers, contexts) -> torch.Tensor:

        objects_from_context = self.choose_object_from_context(contexts)

        has_name_for_object = (
            self.state[speakers, objects_from_context, :, 1].sum(-1) > 0
        )

        if (~has_name_for_object).any():
            n_words = (~has_name_for_object).sum().item()
            words = self.gen_words(n_words)

            self.unique_words.update(words.tolist())

            self.state[
                speakers[~has_name_for_object],
                objects_from_context[~has_name_for_object],
                0,
            ] = torch.stack(
                [words, torch.ones_like(words)],
                dim=-1,
            ).long()

        memory_idx = self.state[speakers, objects_from_context, :, 1].argmax(-1)

        words = self.state[speakers, objects_from_context, memory_idx, 0]

        # Apply random bit flip with probability flip_prob
        if self.flip_prob > 0.0:
            flip_mask = torch.rand(words.shape[0], device=self.device) < self.flip_prob

            if flip_mask.any():
                flipped = self.flip_random_bit(words[flip_mask])

                self.unique_words.update(flipped.tolist())

                words = words.clone()
                words[flip_mask] = flipped

        return words
