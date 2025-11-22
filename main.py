import torch
from utils import get_default_device
from game import Game


def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = Game(10, 10)
    game.play(max_iters=3000)


if __name__ == "__main__":
    main()
