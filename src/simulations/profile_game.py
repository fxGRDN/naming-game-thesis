import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from games.base_game import BaseGame
from parameters import DefaultParams
from utils import get_default_device


def profile_game():
    device = get_default_device()
    
    game = BaseGame(
        game_instances=1000,
        agents=DefaultParams.POPULATION_SIZE.value,
        objects=DefaultParams.OBJECTS_SIZE.value,
        memory=DefaultParams.MEMORY_SIZE.value,
        device=device,
        vocab_size=DefaultParams.VOCAB_SIZE.value,
        context_size=DefaultParams.CONTEXT_SIZE.value,
    )
    
    # Warmup - let CUDA initialize and JIT compile
    print("Warming up...")
    for _ in range(10):
        game.step(0)
    
    torch.cuda.synchronize()
    
    # Profile 100 steps
    print("Profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(100):
            game.step(i)
        torch.cuda.synchronize()
    
    # Print top operations by CUDA time
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Print top operations by CPU time
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CPU TIME")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Print memory usage
    print("\n" + "="*80)
    print("TOP 10 OPERATIONS BY MEMORY")
    print("="*80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    

if __name__ == "__main__":
    profile_game()
