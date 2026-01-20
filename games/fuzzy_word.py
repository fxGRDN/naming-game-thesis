import torch
from .base_game import BaseGame


class FuzzyWordGame(BaseGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**15,
        prune_step=100,
        context_size=(2, 3),
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
            context_size=context_size,
        )
        self.fig_prefix = "fuzzy_word_game"

        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1].")
        self.flip_prob = flip_prob
        # Number of bits needed to represent vocab_size
        self.word_bits = (vocab_size - 1).bit_length()

    def flip_random_bit(self, words: torch.Tensor) -> torch.Tensor:
        bit_pos = torch.randint(
            0, self.word_bits, (words.shape[0],), device=self.device
        )
        flip_mask = 1 << bit_pos
        return words ^ flip_mask

    def communication_channel(self, words):
        if self.flip_prob > 0.0:
            flip_mask = torch.rand(words.shape[0], device=self.device) < self.flip_prob

            if flip_mask.any():
                flipped = self.flip_random_bit(words[flip_mask])

                words = words.clone()
                words[flip_mask] = flipped.to(words.dtype)
        return words
