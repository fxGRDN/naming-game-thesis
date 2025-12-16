import torch
from utils import get_default_device
from base_game import BaseGame


def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = BaseGame(1, 4, 4, device=device)
    game.play(3000)
    game.plot_stats()


if __name__ == "__main__":
    main()
