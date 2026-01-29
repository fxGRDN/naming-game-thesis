import sys
import os
import argparse
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from games.base_game import BaseGame
from parameters import DefaultParams


def profile_game():
    n_games = 1000
    n_steps = 10000
    
    print(f"Profiling {n_games} games, {n_steps} steps each\n")
    
    # CUDA timing
    if torch.cuda.is_available():
        device = torch.device("cuda")
        game = BaseGame(
            game_instances=n_games,
            agents=DefaultParams.POPULATION_SIZE.value,
            objects=DefaultParams.OBJECTS_SIZE.value,
            memory=DefaultParams.MEMORY_SIZE.value,
            device=device,
            vocab_size=DefaultParams.VOCAB_SIZE.value,
            context_size=DefaultParams.CONTEXT_SIZE.value,
        )
        
        # Warmup
        for _ in range(10):
            game.step(0)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for i in range(n_steps):
            game.step(i)
        end_event.record()
        torch.cuda.synchronize()
        
        cuda_time_ms = start_event.elapsed_time(end_event)
        print(f"CUDA: {cuda_time_ms/1000:.2f} s ({cuda_time_ms/n_steps:.3f} ms/step)")
        
        del game
        torch.cuda.empty_cache()
    
    # CPU timing
    device = torch.device("cpu")
    game = BaseGame(
        game_instances=n_games,
        agents=DefaultParams.POPULATION_SIZE.value,
        objects=DefaultParams.OBJECTS_SIZE.value,
        memory=DefaultParams.MEMORY_SIZE.value,
        device=device,
        vocab_size=DefaultParams.VOCAB_SIZE.value,
        context_size=DefaultParams.CONTEXT_SIZE.value,
    )
    
    # Warmup
    for _ in range(10):
        game.step(0)
    
    start = time.perf_counter()
    for i in range(n_steps):
        game.step(i)
    end = time.perf_counter()
    
    cpu_time_ms = (end - start) * 1000
    print(f"CPU:  {cpu_time_ms/1000:.2f} s ({cpu_time_ms/n_steps:.3f} ms/step)")
    
    if torch.cuda.is_available():
        print(f"\nSpeedup: {cpu_time_ms/cuda_time_ms:.1f}x")


if __name__ == "__main__":
    profile_game()
