import torch
from utils import get_default_device
from game import BaseGame


def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = BaseGame(1, 4, 4)
    game.play(1)


if __name__ == "__main__":
    main()
