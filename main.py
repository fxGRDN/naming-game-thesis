import torch
from utils import get_default_device
from games.base_game import BaseGame
from games.fuzzy_object import FuzzyObjectGame
from games.fuzzy_word import FuzzyWordGame
from games.fuzzy_word_object import FuzzyObjectWordGame
import numpy as np
from tqdm import tqdm
import traceback
import time
import os
import tqdm


DEFAULT_POPULATION_SIZE = 16
DEFAULT_OBJECTS_SIZE = 16
DEFAULT_MEMORY_SIZE = 8
DEFAULT_CONTEXT_SIZE = (2, 3)
DEFAULT_VOCAB_SIZE = 2**8


def test_games():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    # game = BaseGame(
    #     game_instances=1000,
    #     agents=16,
    #     objects=16,
    #     memory=8,
    #     device=device,
    #     context_size=(2, 3),
    # )

    # game.play(50000)
    # game.plot_stats()

    fuzzy_object_game = FuzzyObjectGame(
        1000,
        16,
        16,
        memory=8,
        device=device,
        confusion_prob=0.8,
        context_size=(2, 3),
    )

    fuzzy_object_game.play(50000)

    # fuzzy_word_game = FuzzyWordGame(
    #     1000,
    #     32,
    #     16,
    #     prune_step=50,
    #     memory=50,
    #     device=device,
    #     flip_prob=0.3,
    #     context_size=(2, 3),
    # )

    # fuzzy_word_game.play(50000)
    # fuzzy_word_game.plot_stats()

    # fuzzy_object_word_game = FuzzyObjectWordGame(
    #     game_instances=1,
    #     agents=16,
    #     objects=16,
    #     memory=10,
    #     vocab_size=2**8,
    #     confusion_prob=0.5,
    #     flip_prob=0.5,
    #     device=device,
    # )

    # fuzzy_object_word_game.play(1000)
    # fuzzy_object_word_game.plot_stats()


# def monte_carlo_simulation():
#     # Example placeholder for Monte Carlo simulation logic
#     num_simulations = 2

#     p_linspace = tqdm(np.linspace(0, 1, 10), position=0)
#     q_linspace = np.linspace(0, 1, 10)
#     sims = range(num_simulations)

#     results = np.zeros((len(q_linspace), len(p_linspace), num_simulations))
#     try:
#         for i, q in enumerate(q_linspace):
#             for j, p in enumerate(p_linspace):
#                 for k in sims:
#                     device: torch.device = get_default_device()
#                     game = FuzzyObjectWordGame(
#                         game_instances=1,
#                         agents=50,
#                         objects=50,
#                         memory=20,
#                         vocab_size=2**10,
#                         confusion_prob=p,
#                         flip_prob=q,
#                         device=device,
#                     )
#                     game.play(1000)
#                     results[i, j, k] = game.stats[0, -1]

#         np.save("monte_carlo_results.npy", results)
#     except Exception as e:, max_agent_pairs=1
#         with open("error_log.txt", "w") as f:
#             f.write(f"An error occurred: {e}\n\n")
#             f.write("Traceback:\n")
#             f.write(traceback.format_exc())
#         print(f"An error occurred: {e} (see error_log.txt for details)")




def test_base_game():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 10000
    stats = np.zeros((iters, 4, game_steps))

    game = BaseGame(
            game_instances=iters,
            agents=DEFAULT_POPULATION_SIZE, 
            objects=DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE, 
            context_size=DEFAULT_CONTEXT_SIZE,
        )
    game.play(game_steps, tqdm_desc="Base Game Test Simulation")
    game.plot_stats()


def baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 50000
    game = BaseGame(
            game_instances=iters,
            agents=DEFAULT_POPULATION_SIZE, 
            objects=DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE, 
            context_size=DEFAULT_CONTEXT_SIZE,
        )
    game.play(game_steps, tqdm_desc="Baseline Simulation")


    stats = game.stats.cpu().numpy()
    np.save("data/base_game_monte_carlo_stats.npy", stats)

def population_size_base():
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 100000
    population_sizes = [8, 16, 32, 64]
    stats = np.zeros((len(population_sizes), 4, game_steps, games))

    os.makedirs("data/population_size_state", exist_ok=True)

    for j, pop_size in enumerate(population_sizes):
        game = BaseGame(
            games,
            pop_size, 
            DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_size=DEFAULT_CONTEXT_SIZE,
        )
        game.play(game_steps, tqdm_desc=f"Population Size {pop_size}")

        stats[j] = game.stats.cpu().numpy()
        np.save(f"data/population_size_state/{pop_size}.npy", game.state.cpu().numpy())



def object_size_base():
    print("Starting object size consensus simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 50000
    object_sizes = [8, 16, 32, 64, 128, 256]
    stats = np.zeros((len(object_sizes), 4, game_steps, games))

    for j, obj_size in enumerate(object_sizes):
        game = BaseGame(
            games,
            DEFAULT_POPULATION_SIZE, 
            obj_size, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_size=DEFAULT_CONTEXT_SIZE,
        )
        game.play(game_steps, tqdm_desc=f"Object Size {obj_size}", calc_stats=False)

        stats[j] = game.stats.cpu().numpy()
    np.save("data/base_game_object_size_base.npy", stats)


def memory_size_base():
    print("Starting memory size base simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 50000
    memory_sizes = [1, 2, 4, 8, 16]
    stats = np.zeros((len(memory_sizes), 4, game_steps, games))

    for j, mem_size in enumerate(memory_sizes):
        game = BaseGame(
            games,
            DEFAULT_POPULATION_SIZE, 
            DEFAULT_OBJECTS_SIZE, 
            memory=mem_size, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_size=DEFAULT_CONTEXT_SIZE,
        )
        game.play(game_steps, tqdm_desc=f"Memory Size {mem_size}")

        stats[j] = game.stats.cpu().numpy()
    np.save("data/base_game_memory_size_base.npy", stats)


def vocab_size_base():
    print("Starting vocab size base simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 50000
    vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]
    stats = np.zeros((len(vocab_sizes), 4, game_steps, games))

    for j, vocab_size in enumerate(vocab_sizes):
        game = BaseGame(
            games,
            DEFAULT_POPULATION_SIZE, 
            DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=vocab_size,
            context_size=DEFAULT_CONTEXT_SIZE,
        )
        game.play(game_steps, tqdm_desc=f"Vocab Size {vocab_size}")

        stats[j] = game.stats.cpu().numpy()
    np.save("data/base_game_vocab_size_base.npy", stats)

def context_window_size_base():
    print("Starting context window size base simulation...")
    device: torch.device = get_default_device()
    games = 1000
    game_steps = 50000
    context_sizes = [(2, 2), (2, 3), (3, 4), (4, 5)]
    stats = np.zeros((len(context_sizes), 4, game_steps, games))

    for j, context_size in enumerate(context_sizes):
        game = BaseGame(
            games,
            DEFAULT_POPULATION_SIZE, 
            DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_VOCAB_SIZE,
            context_size=context_size,
        )
        game.play(game_steps, tqdm_desc=f"Context Size {context_size}")

        stats[j] = game.stats.cpu().numpy()
    np.save("data/base_game_context_size_base.npy", stats)


def word_baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    bit_flip_prob = np.linspace(0, 1, 100)

    for i, p in enumerate(bit_flip_prob):
        game = FuzzyWordGame(
                game_instances=iters,
                agents=DEFAULT_POPULATION_SIZE, 
                objects=DEFAULT_OBJECTS_SIZE, 
                memory=DEFAULT_MEMORY_SIZE, 
                device=device, 
                flip_prob=p,
                vocab_size=DEFAULT_VOCAB_SIZE, 
                context_size=DEFAULT_CONTEXT_SIZE,
            )
        game.play(game_steps, disable_tqdm=True)

        np.save(f"data/word_phase/part_{i}.npy", game.stats.cpu().numpy())

def object_baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 100000
    obj_conf = np.linspace(0, 1, 100)

    os.makedirs("data/object_phase", exist_ok=True)

    for i, conf in tqdm.tqdm(enumerate(obj_conf), desc="Object Baseline Simulation", total=len(obj_conf)):
        game = FuzzyObjectGame(
                game_instances=iters,
                agents=DEFAULT_POPULATION_SIZE, 
                objects=DEFAULT_OBJECTS_SIZE, 
                memory=DEFAULT_MEMORY_SIZE, 
                device=device, 
                confusion_prob=conf,
                vocab_size=DEFAULT_VOCAB_SIZE, 
                context_size=DEFAULT_CONTEXT_SIZE,
            )
        game.play(game_steps, tqdm_desc=f"Object Simulation for p={conf}", disable_tqdm=True)
        np.save(f"data/object_phase/part_{i}.npy", game.stats.cpu().numpy())


def word_object_baseline():
    device: torch.device = get_default_device()
    iters = 1000
    game_steps = 50000
    obj_conf = np.linspace(0, 1, 30)
    bit_flip_prob = np.linspace(0, 1, 30)

    os.makedirs("data/word_object_game", exist_ok=True)


    for i, conf in enumerate(obj_conf):
        time_start = time.time()
        stats = np.zeros((len(bit_flip_prob), 4, game_steps, iters))
        for j, flip_prob in tqdm.tqdm(enumerate(bit_flip_prob)):
            game = FuzzyObjectWordGame(
                    game_instances=iters,
                    agents=DEFAULT_POPULATION_SIZE, 
                    objects=DEFAULT_OBJECTS_SIZE, 
                    memory=DEFAULT_MEMORY_SIZE, 
                    device=device, 
                    confusion_prob=conf,
                    flip_prob=flip_prob,
                    vocab_size=DEFAULT_VOCAB_SIZE, 
                    context_size=DEFAULT_CONTEXT_SIZE,
                )
            game.play(game_steps, tqdm_desc="Word-Object Simulation", disable_tqdm=True)

            stats[j] = game.stats.cpu().numpy()
        print(f"Saved part {i+25} of {len(obj_conf)}, time taken: {(time.time() - time_start)/60:.2f} minutes")
        np.save(f"data/word_object_game/monte_carlo_stats_part_{i+25}.npy", stats)



def clusters():
    device: torch.device = get_default_device()
    iters = 1
    game_steps = 1000000
    snapshot_state = [0, 10, 100, 1000, 10000, 100000, 500000, 999999]

    game = BaseGame(
            game_instances=iters,
            agents=512, 
            objects=DEFAULT_OBJECTS_SIZE, 
            memory=DEFAULT_MEMORY_SIZE, 
            device=device, 
            vocab_size=DEFAULT_OBJECTS_SIZE, 
            context_size=DEFAULT_CONTEXT_SIZE,
        )
    game.play(game_steps, tqdm_desc="Vocab Size Analysis", snapshot_state=snapshot_state, snap_name="classic_big_pop")

if __name__ == "__main__":
    try:
        # test_games()
        # baseline()
        # population_size_base()
        # object_size_base()
        # memory_size_base()
        # vocab_size_base()
        # context_window_size_base()
        word_baseline()
        # object_baseline()
        # word_object_baseline()
        # clusters()
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(f"An error occurred: {e}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        print(f"An error occurred: {e} (see error_log.txt for details)")
