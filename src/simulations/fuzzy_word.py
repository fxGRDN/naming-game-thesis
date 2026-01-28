import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from utils import get_default_device
from games.fuzzy_word import FuzzyWordGame
from parameters import DefaultParams


def word_baseline(args):
    print(f"Starting word baseline simulation...")
    print(f"Sampling interval: {args.sampling_freq}")
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    bit_flip_prob = np.linspace(0, 1, 100)

    for part_idx, p in enumerate(tqdm(bit_flip_prob, desc="Bit Flip Probabilities")):
        game = FuzzyWordGame(
                game_instances=iters,
                agents=args.population_size,
                objects=args.object_size, 
                memory=args.memory_size, 
                device=device, 
                flip_prob=p,
                vocab_size=args.vocab_size, 
                context_size=tuple(args.context_size) if args.context_size else DefaultParams.CONTEXT_SIZE.value,
            )
        game.play(game_steps, disable_tqdm=True, sampling_freq=args.sampling_freq)

    np.save(f"data/word_phase/part_{part_idx}.npy", game.stats.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    parser.add_argument("-p", "--population-size", type=int, default=DefaultParams.POPULATION_SIZE.value, help="Population size for the simulation")
    parser.add_argument("-o", "--object-size", type=int, default=DefaultParams.OBJECTS_SIZE.value, help="Object size for the simulation")   
    parser.add_argument("-m", "--memory-size", type=int, default=DefaultParams.MEMORY_SIZE.value, help="Memory size for the simulation")
    parser.add_argument("-v", "--vocab-size", type=int, default=DefaultParams.VOCAB_SIZE.value, help="Vocabulary size for the simulation")
    parser.add_argument("-c", "--context-size", nargs=2, type=int, metavar=("MIN","MAX"), help="Context size range for the simulation")
    args = parser.parse_args()
    

    os.makedirs("data/word_phase", exist_ok=True)
    word_baseline(args)