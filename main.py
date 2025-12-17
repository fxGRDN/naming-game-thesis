import torch
from utils import get_default_device
from games.base_game import BaseGame
from games.fuzzy_object import FuzzyObjectGame
from games.fuzzy_word import FuzzyWordGame
from games.fuzzy_word_object import FuzzyObjectWordGame
import numpy as np
from tqdm import tqdm


def test_games():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = BaseGame(1, 16, 16, memory=10, device=device, vocab_size=2**8)
    game.play(1000)
    game.plot_stats()

    fuzzy_object_game = FuzzyObjectGame(
        game_instances=1,
        agents=16,
        objects=16,
        memory=10,
        vocab_size=2**8,
        confusion_prob=0.5,
        device=device,
    )

    fuzzy_object_game.play(1000)
    fuzzy_object_game.plot_stats()

    fuzzy_word_game = FuzzyWordGame(
        game_instances=1,
        agents=16,
        objects=16,
        memory=10,
        vocab_size=2**8,
        flip_prob=0.5,
        device=device,
    )

    fuzzy_word_game.play(1000)
    fuzzy_word_game.plot_stats()

    fuzzy_object_word_game = FuzzyObjectWordGame(
        game_instances=1,
        agents=16,
        objects=16,
        memory=10,
        vocab_size=2**8,
        confusion_prob=0.5,
        flip_prob=0.5,
        device=device,
    )

    fuzzy_object_word_game.play(1000)
    fuzzy_object_word_game.plot_stats()


def monte_carlo_simulation():
    # Example placeholder for Monte Carlo simulation logic
    num_simulations = 100

    p_linspace = tqdm(np.linspace(0, 1, 100), position=0)
    q_linspace = tqdm(np.linspace(0, 1, 100), position=1)
    sims = tqdm(range(num_simulations), position=2)

    results = np.zeros((len(q_linspace), len(p_linspace), num_simulations))
    for i, q in enumerate(q_linspace):
        for j, p in enumerate(p_linspace):
            for k in sims:
                device: torch.device = get_default_device()
                game = FuzzyObjectWordGame(
                    game_instances=1,
                    agents=50,
                    objects=50,
                    memory=5,
                    vocab_size=2**10,
                    confusion_prob=p,
                    flip_prob=q,
                    device=device,
                )
                game.play(1000)
                results[i, j, k] = game.stats[0, -1]

    np.save("monte_carlo_results.npy", results)


if __name__ == "__main__":
    # main()
    monte_carlo_simulation()
