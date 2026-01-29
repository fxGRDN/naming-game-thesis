
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



def object_baseline(args):
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    obj_conf = np.linspace(0, 1, 100)

    os.makedirs("data/object_phase", exist_ok=True)

    for i, conf in tqdm.tqdm(enumerate(obj_conf), desc="Object Baseline Simulation", total=len(obj_conf)):
        game = FuzzyObjectGame(
                game_instances=iters,
                agents=args.population_size, 
                objects=args.object_size, 
                memory=args.memory_size, 
                device=device, 
                confusion_prob=conf,
                vocab_size=args.vocab_size, 
                context_size=tuple(args.context_size) if args.context_size else DefaultParams.CONTEXT_SIZE.value,
            )
        game.play(game_steps, disable_tqdm=True, sampling_freq=args.sampling_freq)
        np.save(f"data/object_phase/part_{i}.npy", game.stats.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    parser.add_argument("-p", "--population-size", type=int, default=DefaultParams.POPULATION_SIZE.value, help="Population size for the simulation")
    parser.add_argument("-o", "--object-size", type=int, default=DefaultParams.OBJECTS_SIZE.value, help="Object size for the simulation")   
    parser.add_argument("-m", "--memory-size", type=int, default=DefaultParams.MEMORY_SIZE.value, help="Memory size for the simulation")
    parser.add_argument("-v", "--vocab-size", type=int, default=DefaultParams.VOCAB_SIZE.value, help="Vocabulary size for the simulation")
    parser.add_argument("-c", "--context-size", nargs=2, type=int, metavar=("MIN","MAX"), help="Context size range for the simulation")
    args = parser.parse_args()
    
    os.makedirs("data/object_phase", exist_ok=True)
    object_baseline(args)