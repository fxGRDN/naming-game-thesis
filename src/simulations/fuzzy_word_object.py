import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tqdm
from utils import get_default_device
from games.fuzzy_word_object import FuzzyObjectWordGame
from parameters import DefaultParams





def word_object_baseline(sampling_freq: int):
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    obj_conf = np.linspace(0, 1, 25)
    bit_flip_prob = np.linspace(0, 1, 25)

    os.makedirs("data/word_object_game", exist_ok=True)


    for i, conf in tqdm.tqdm(enumerate(obj_conf), desc="Word-Object Baseline Simulation", total=len(obj_conf)):
        stats = np.zeros((len(bit_flip_prob), 4, game_steps // sampling_freq, iters))
        for j, flip_prob in enumerate(bit_flip_prob):
            game = FuzzyObjectWordGame(
                    game_instances=iters,
                    agents=DefaultParams.POPULATION_SIZE.value, 
                    objects=DefaultParams.OBJECTS_SIZE.value, 
                    memory=8, 
                    device=device, 
                    confusion_prob=conf,
                    flip_prob=flip_prob,
                    vocab_size=DefaultParams.VOCAB_SIZE.value, 
                    context_size=DefaultParams.CONTEXT_SIZE.value,
                )
            game.play(game_steps, tqdm_desc="Word-Object Simulation", disable_tqdm=True, sampling_freq=sampling_freq)

            stats[j] = game.stats.cpu().numpy()
        np.save(f"data/word_object_game/monte_carlo_stats_part_{i}.npy", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    args = parser.parse_args()
    os.makedirs("data/object_phase", exist_ok=True)
    word_object_baseline(args.sampling_freq)