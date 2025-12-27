import torch
from utils import get_default_device
from games.base_game import BaseGame
from games.fuzzy_object import FuzzyObjectGame
from games.fuzzy_word import FuzzyWordGame
from games.fuzzy_word_object import FuzzyObjectWordGame
import numpy as np
from tqdm import tqdm
import traceback


def test_games():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = BaseGame(
        1,
        16,
        16,
        memory=50,
        device=device,
        context_size=(2, 3),
    )

    game.play(10000)
    game.plot_stats()

    # fuzzy_object_game = FuzzyObjectGame(
    #     game_instances=1,
    #     agents=16,
    #     objects=16,
    #     memory=10,
    #     vocab_size=2**8,
    #     confusion_prob=0.5,
    #     device=device,
    # )

    # fuzzy_object_game.play(1000)
    # fuzzy_object_game.plot_stats()

    # fuzzy_word_game = FuzzyWordGame(
    #     game_instances=1,
    #     agents=16,
    #     objects=16,
    #     memory=10,
    #     vocab_size=2**8,
    #     flip_prob=0.5,
    #     device=device,
    # )

    # fuzzy_word_game.play(1000)
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
#     except Exception as e:
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

    for i in tqdm(range(iters), desc="Monte Carlo Simulations"):
        game = BaseGame(
            16, 16, memory=20, device=device, vocab_size=2**8, max_agent_pairs=1
        )
        game.play(game_steps)

        stats[i] = game.stats.cpu().numpy()
    np.save("data/base_game_monte_carlo_stats_one_pair.npy", stats)

    game_steps = 1000

    stats = np.zeros((iters, 4, game_steps))

    for i in tqdm(range(iters), desc="Monte Carlo Simulations"):
        game = BaseGame(16, 16, memory=20, device=device, vocab_size=2**8)
        game.play(game_steps)

        stats[i] = game.stats.cpu().numpy()
    np.save("data/base_game_monte_carlo_stats_all_pairs.npy", stats)


if __name__ == "__main__":
    try:
        test_games()
        # monte_carlo_simulation()
        # test_base_game()
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(f"An error occurred: {e}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        print(f"An error occurred: {e} (see error_log.txt for details)")
