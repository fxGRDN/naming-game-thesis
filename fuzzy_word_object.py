from fuzzy_object import FuzzyObjectGame
from fuzzy_word import FuzzyWordGame
import torch


class FuzzyObjectWordGame(FuzzyObjectGame, FuzzyWordGame):
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**8,
        prune_step=100,
        confusion_prob=0.0,  # p
        flip_prob=0.0,  # q
        device=torch.device("cpu"),
    ) -> None:

        # Call FuzzyObjectGame's init (which calls BaseGame)
        FuzzyObjectGame.__init__(
            self,
            game_instances=game_instances,
            agents=agents,
            objects=objects,
            memory=memory,
            vocab_size=vocab_size,
            prune_step=prune_step,
            confusion_prob=confusion_prob,
            device=device,
        )

        # Set FuzzyWordGame-specific attributes
        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1].")
        self.flip_prob = flip_prob
        self.word_bits = (vocab_size - 1).bit_length()
        self.fig_prefix = "fuzzy_object_word_game"
