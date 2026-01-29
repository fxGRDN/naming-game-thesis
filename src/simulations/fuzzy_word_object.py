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





def word_object_baseline(args):
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    obj_conf = np.linspace(0, 1, 25)
    bit_flip_prob = np.linspace(0, 1, 25)

    os.makedirs("data/word_object_game", exist_ok=True)


    for i, conf in tqdm.tqdm(enumerate(obj_conf), desc="Word-Object Baseline Simulation", total=len(obj_conf)):
        stats = np.zeros((len(bit_flip_prob), 4, game_steps // args.sampling_freq, iters))
        for j, flip_prob in enumerate(bit_flip_prob):
            game = FuzzyObjectWordGame(
                    game_instances=iters,
                    agents=args.population_size, 
                    objects=args.object_size, 
                    memory=args.memory_size, 
                    device=device, 
                    confusion_prob=conf,
                    flip_prob=flip_prob,
                    vocab_size=args.vocab_size, 
                    context_size=tuple(args.context_size) if args.context_size else DefaultParams.CONTEXT_SIZE.value,
                )
            game.play(game_steps, tqdm_desc="Word-Object Simulation", disable_tqdm=True, sampling_freq=args.sampling_freq)

            stats[j] = game.stats.cpu().numpy()
        np.save(f"data/word_object_game/monte_carlo_stats_part_{i}.npy", stats)


def word_object_single(p, q, args):
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 1000000


    os.makedirs("data/word_object_game/single", exist_ok=True)

    game = FuzzyObjectWordGame(
            game_instances=iters,
            agents=args.population_size, 
            objects=args.object_size, 
            memory=args.memory_size, 
            device=device, 
            confusion_prob=p,
            flip_prob=q,
            vocab_size=args.vocab_size, 
            context_size=tuple(args.context_size) if args.context_size else DefaultParams.CONTEXT_SIZE.value,
        )
    game.play(game_steps, tqdm_desc="Word-Object Simulation", sampling_freq=args.sampling_freq)

    np.save(f"data/word_object_game/single/monte_carlo_stats_part_{p}_{q}.npy", game.stats.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    parser.add_argument("-p", "--population-size", type=int, default=DefaultParams.POPULATION_SIZE.value, help="Population size for the simulation")
    parser.add_argument("-o", "--object-size", type=int, default=DefaultParams.OBJECTS_SIZE.value, help="Object size for the simulation")   
    parser.add_argument("-m", "--memory-size", type=int, default=DefaultParams.MEMORY_SIZE.value, help="Memory size for the simulation")
    parser.add_argument("-v", "--vocab-size", type=int, default=DefaultParams.VOCAB_SIZE.value, help="Vocabulary size for the simulation")
    parser.add_argument("-c", "--context-size", nargs=2, type=int, metavar=("MIN","MAX"), help="Context size range for the simulation")

    parser.add_argument("--single", action="store_true", help="Run single simulation instead of monte-carlo sweep")  
    parser.add_argument("-p", "--confusion-prob", type=float, help="Object confusion probability for single run", default=None)
    parser.add_argument("-q", "--flip-prob", type=float, help="Word bit flip probability for single run", default=None)
    args = parser.parse_args()
    os.makedirs("data/word_object_game/single", exist_ok=True)

    if args.single:
        if args.confusion_prob is None or args.flip_prob is None:
            raise ValueError("For single run, both --confusion-prob and --flip-prob must be specified.")
        word_object_single(args.confusion_prob, args.flip_prob, args)
    else:               
        word_object_baseline(args)