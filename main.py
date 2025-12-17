import torch
from utils import get_default_device
from game import BaseGame
from fuzzy_object import FuzzyObjectGame
from fuzzy_word import FuzzyWordGame
from fuzzy_word_object import FuzzyObjectWordGame


def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    # game = BaseGame(1, 16, 16, memory=10, device=device, vocab_size=2**8)
    # game.play(1000)
    # game.plot_stats()

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


if __name__ == "__main__":
    main()
