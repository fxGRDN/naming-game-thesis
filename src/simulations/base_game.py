import argparse
import sys
import os

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from games.base_game import BaseGame
from parameters import DefaultParams
from utils import get_default_device


def baseline(sampling_freq: int = 100):
    print("Starting baseline simulation...")
    device: torch.device = get_default_device()
    os.makedirs("data/baseline", exist_ok=True)
    iters = 1000
    game_steps = 50_000
    game = BaseGame(
            game_instances=iters,
            agents=DefaultParams.POPULATION_SIZE.value, 
            objects=DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value, 
            context_size=DefaultParams.CONTEXT_SIZE.value,

        )
    game.play(game_steps, sampling_freq=sampling_freq)


    stats = game.stats.cpu().numpy()
    np.save("data/baseline/baseline.npy", stats)



def population_size_base(sampling_freq: int = 100):
    print("Starting population size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    population_sizes = [8, 16, 32, 64, 128, 256]
    stats = np.zeros((len(population_sizes), 4, game_steps // sampling_freq, games))

    os.makedirs("data/baseline", exist_ok=True)

    for j, pop_size in tqdm(enumerate(population_sizes), desc="Population Size Simulation", total=len(population_sizes)):
        game = BaseGame(
            games,
            pop_size, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, sampling_freq=sampling_freq, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save(f"data/baseline/population_size.npy", stats)



def object_size_base(sampling_freq: int = 100):
    print("Starting object size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    object_sizes = [8, 16, 32, 64, 128, 256]
    stats = np.zeros((len(object_sizes), 4, game_steps // sampling_freq, games))

    for j, obj_size in tqdm(enumerate(object_sizes), desc="Object Size Simulation", total=len(object_sizes)):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            obj_size, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, sampling_freq=sampling_freq, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/object_size.npy", stats)



def vocab_size_base(sampling_freq: int = 100):
    print("Starting vocab size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]
    stats = np.zeros((len(vocab_sizes), 4, game_steps, games))

    for j, vocab_size in tqdm(enumerate(vocab_sizes), desc="Vocab Size Simulation", total=len(vocab_sizes)):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=vocab_size,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, sampling_freq=sampling_freq, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/vocab_size.npy", stats)

def context_window_size_base(sampling_freq: int = 100):
    print("Starting context window size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    context_sizes = [(1, 2), (2, 4), (4, 6), (6, 8)]
    stats = np.zeros((len(context_sizes), 4, game_steps // sampling_freq, games))

    for j, context_size in tqdm(enumerate(context_sizes), desc="Context Window Size Simulation", total=len(context_sizes)):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=context_size,
        )
        game.play(game_steps, sampling_freq=sampling_freq, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/context_size.npy", stats)


def memory_size_base(sampling_freq: int = 100):
    print("Starting memory size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    memory_sizes = [3, 5, 7, 8, 12, 16]
    stats = np.zeros((len(memory_sizes), 4, game_steps // sampling_freq, games))

    for j, memory_size in tqdm(enumerate(memory_sizes), desc="Memory Size Simulation", total=len(memory_sizes)):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=memory_size, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, sampling_freq=sampling_freq, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/memory_size.npy", stats)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", action="store_true", help="Run baseline simulation")
    parser.add_argument("-p", action="store_true", help="Run population size simulation")
    parser.add_argument("-o", action="store_true", help="Run object size simulation")
    parser.add_argument("-v", action="store_true", help="Run vocab size simulation")
    parser.add_argument("-c", action="store_true", help="Run context window size simulation")
    parser.add_argument("-m", action="store_true", help="Run memory size simulation")
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    args = parser.parse_args()
    os.makedirs("data/baseline", exist_ok=True)

    
    if args.b:
        baseline(args.sampling_freq)
    if args.p:
        population_size_base(args.sampling_freq)
    if args.o:
        object_size_base(args.sampling_freq)
    if args.v:
        vocab_size_base(args.sampling_freq)
    if args.c:
        context_window_size_base(args.sampling_freq)
    if args.m:
        memory_size_base(args.sampling_freq)