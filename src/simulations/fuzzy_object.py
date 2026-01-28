
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tqdm
from utils import get_default_device
from games.fuzzy_object import FuzzyObjectGame
from parameters import DefaultParams



def object_baseline(sampling_freq: int):
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    obj_conf = np.linspace(0, 1, 100)

    os.makedirs("data/object_phase", exist_ok=True)

    for i, conf in tqdm.tqdm(enumerate(obj_conf), desc="Object Baseline Simulation", total=len(obj_conf)):
        game = FuzzyObjectGame(
                game_instances=iters,
                agents=DefaultParams.POPULATION_SIZE.value, 
                objects=DefaultParams.OBJECTS_SIZE.value, 
                memory=DefaultParams.MEMORY_SIZE.value, 
                device=device, 
                confusion_prob=conf,
                vocab_size=DefaultParams.VOCAB_SIZE.value, 
                context_size=DefaultParams.CONTEXT_SIZE.value,
            )
        game.play(game_steps, disable_tqdm=True, sampling_freq=sampling_freq)
        np.save(f"data/object_phase/part_{i}.npy", game.stats.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    args = parser.parse_args()
    os.makedirs("data/object_phase", exist_ok=True)
    object_baseline(args.sampling_freq)