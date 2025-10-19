import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class Game:
    def __init__(self, n_players: int, n_objects: int = 10, rounds: int = 10):
        self.n_players = n_players
        self.n_objects = n_objects
        # each player has a matrix of word-object associations
        self.game_state = torch.zeros((n_players, n_objects, n_players), dtype=torch.int32)
        self.rounds = rounds

    def _init_round(self) -> None:
        speakers, hearers = self._sample_players()
        objects = self._sample_objects()

        self.game_state[speakers, objects, speakers] += 1
        self.game_state[hearers, objects, self.game_state[speakers, objects].argmax(dim=-1)] += 1


    def _next_round(self) -> None:
        speakers, hearers = self._sample_players()
        objects = self._sample_objects()

        if not torch.any(self.game_state[speakers, objects]):
            # if speaker has no words for the object, invent a new word
            self.game_state[speakers, objects, speakers] += 1

        self.game_state[hearers, objects, self.game_state[speakers, objects].argmax(dim=-1)] += 1

    def _sample_players(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.n_players // 2
        speaker = torch.randperm(self.n_players, device=self.game_state.device)[:count]
        mask = torch.ones(self.n_players, dtype=torch.bool, device=self.game_state.device)
        mask[speaker] = False
        hearer = mask.nonzero(as_tuple=False).squeeze(-1)
        return speaker, hearer

    def _sample_objects(self) -> torch.Tensor:
        return torch.randint(0, self.n_objects, (self.n_players // 2,), device=self.game_state.device)


    def _object_consensus(self) -> torch.Tensor:
        word_idx = self.game_state.argmax(dim=-1)  # (players, objects)
        consensus = word_idx.mode(dim=0).values  # (objects,)
        return consensus
        
    def _percent_object_consensus(self) -> float:
        consensus = self._object_consensus()  # (objects,)
        word_idx = self.game_state.argmax(dim=-1)  # (players, objects)
        matches = word_idx.eq(consensus.unsqueeze(0))  # (players, objects)
        percent_consensus = matches.float().mean().item()
        return percent_consensus
    
    def play_game(self):
        self._init_round()
        consensus_over_time: list[float] = []
        for _ in range(self.rounds - 1):
            self._next_round()
            cons: float = self._percent_object_consensus()
            consensus_over_time.append(cons)
            print(cons)

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots() # type: ignore
        ax.plot(range(1, self.rounds), consensus_over_time)
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Percent Object Consensus")
        ax.set_title("Object Consensus Over Time")
        ax.set_ylim(0, 1)
        ax.grid(True)
        fig.savefig("object_consensus_over_time.png")
