import torch
from utils import get_default_device
from game import Game

def main():
    device: torch.device = get_default_device()
    print(f"Using device: {device}")

    game = Game(n_players=100, n_objects=100,rounds=1000)
    game.play_game()

    


if __name__ == "__main__":
    main()
