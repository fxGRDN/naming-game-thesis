import torch
from utils import get_default_device
from game import BaseGame


def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = BaseGame(1, 16, 16, memory=10, device=device, vocab_size=2**8)
    game.play(1000)
    game.plot_stats()


if __name__ == "__main__":
    main()
