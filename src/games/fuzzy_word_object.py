from .fuzzy_object import FuzzyObjectGame
from .fuzzy_word import FuzzyWordGame
import torch


class FuzzyObjectWordGame(FuzzyObjectGame, FuzzyWordGame):
    
    def __init__(
        self,
        game_instances: int = 1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**8,
        confusion_prob=0.0,
        flip_prob=0.0,
        context_size: tuple[int, int] = (2, 3),
        device=torch.device("cpu"),
    ) -> None:
        super().__init__(
            game_instances=game_instances,
            agents=agents,
            objects=objects,
            memory=memory,
            vocab_size=vocab_size,
            confusion_prob=confusion_prob,
            context_size=context_size,
            device=device,
        )
        
        if not (0.0 <= float(flip_prob) <= 1.0):
            raise ValueError("flip_prob must be in [0, 1].")
        self.flip_prob = torch.as_tensor(flip_prob, device=device, dtype=torch.float32)
        self.word_bits = (vocab_size - 1).bit_length()
        self.fig_prefix = "fuzzy_object_word_game"
