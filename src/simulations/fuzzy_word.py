import numpy as np
import os
import torch
from utils import get_default_device
from games.fuzzy_word import FuzzyWordGame
from parameters import DefaultParams


def word_baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    bit_flip_prob = np.linspace(0, 1, 100)

    for i, p in enumerate(bit_flip_prob):
        game = FuzzyWordGame(
                game_instances=iters,
                agents=DefaultParams.POPULATION_SIZE.value, 
                objects=DefaultParams.OBJECTS_SIZE.value, 
                memory=DefaultParams.MEMORY_SIZE.value, 
                device=device, 
                flip_prob=p,
                vocab_size=DefaultParams.VOCAB_SIZE.value, 
                context_size=DefaultParams.CONTEXT_SIZE.value,
            )
        game.play(game_steps, disable_tqdm=True)

        np.save(f"data/word_phase/part_{i}.npy", game.stats.cpu().numpy())