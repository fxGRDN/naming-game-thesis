import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from games.base_game import BaseGame
from parameters import DefaultParams
from utils import get_default_device


def baseline():
    print("Starting baseline simulation...")
    device: torch.device = get_default_device()
    os.makedirs("data/baseline", exist_ok=True)
    iters = 1000
    game_steps = 50000
    game = BaseGame(
            game_instances=iters,
            agents=DefaultParams.POPULATION_SIZE.value, 
            objects=DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value, 
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
    game.play(game_steps, disable_tqdm=True)


    stats = game.stats.cpu().numpy()
    np.save("data/baseline/baseline.npy", stats)



def population_size_base():
    print("Starting population size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    population_sizes = [8, 16, 32, 64, 128, 256]
    stats = np.zeros((len(population_sizes), 4, game_steps, games))

    os.makedirs("data/baseline", exist_ok=True)

    for j, pop_size in enumerate(population_sizes):
        game = BaseGame(
            games,
            pop_size, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save(f"data/baseline/population_size.npy", stats)



def object_size_base():
    print("Starting object size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    object_sizes = [8, 16, 32, 64, 128, 256]
    stats = np.zeros((len(object_sizes), 4, game_steps, games))

    for j, obj_size in enumerate(object_sizes):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            obj_size, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/object_size.npy", stats)



def vocab_size_base():
    print("Starting vocab size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]
    stats = np.zeros((len(vocab_sizes), 4, game_steps, games))

    for j, vocab_size in enumerate(vocab_sizes):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=vocab_size,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        game.play(game_steps, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/vocab_size.npy", stats)

def context_window_size_base():
    print("Starting context window size simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    context_sizes = [(1, 2), (2, 4), (4, 6), (6, 8)]
    stats = np.zeros((len(context_sizes), 4, game_steps, games))

    for j, context_size in enumerate(context_sizes):
        game = BaseGame(
            games,
            DefaultParams.POPULATION_SIZE.value, 
            DefaultParams.OBJECTS_SIZE.value, 
            memory=DefaultParams.MEMORY_SIZE.value, 
            device=device, 
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=context_size,
        )
        game.play(game_steps, disable_tqdm=True)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/baseline/context_size.npy", stats)


baseline()
population_size_base()
object_size_base()
vocab_size_base()
context_window_size_base()