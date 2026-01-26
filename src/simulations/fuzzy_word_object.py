import torch
import time
import numpy as np
import os
import tqdm
from utils import get_default_device
from games.fuzzy_word_object import FuzzyObjectWordGame
from parameters import DefaultParams





def word_object_baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 50000
    obj_conf = np.linspace(0, 0.7, 25)
    bit_flip_prob = np.linspace(0, 0.7, 25)

    os.makedirs("data/word_object_game", exist_ok=True)


    for i, conf in enumerate(obj_conf):
        time_start = time.time()
        stats = np.zeros((len(bit_flip_prob), 4, game_steps, iters))
        for j, flip_prob in tqdm.tqdm(enumerate(bit_flip_prob)):
            game = FuzzyObjectWordGame(
                    game_instances=iters,
                    agents=DefaultParams.POPULATION_SIZE.value, 
                    objects=DefaultParams.OBJECTS_SIZE.value, 
                    memory=DefaultParams.MEMORY_SIZE.value, 
                    device=device, 
                    confusion_prob=conf,
                    flip_prob=flip_prob,
                    vocab_size=DefaultParams.VOCAB_SIZE.value, 
                    context_size=DefaultParams.CONTEXT_SIZE.value,
                )
            game.play(game_steps, tqdm_desc="Word-Object Simulation", disable_tqdm=True)

            stats[j] = game.stats.cpu().numpy()
        print(f"Saved part {i} of {len(obj_conf)}, time taken: {(time.time() - time_start)/60:.2f} minutes")
        np.save(f"data/word_object_game/monte_carlo_stats_part_{i}.npy", stats)


