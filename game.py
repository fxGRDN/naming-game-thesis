import numpy as np
from functools import reduce
import random
import string
import matplotlib.pyplot as plt

type Vocabulary = dict[int, dict[str, int]]


def max_word(x: tuple[str, int], y: tuple[str, int]):
    return x if x[1] > y[1] else y


def sum_count(x: tuple[str, int], y: tuple[str, int]):
    return ("", x[1] + y[1])


class Player:

    def __init__(self, word_length=2) -> None:
        self.vocabulary: Vocabulary = {}
        self.word_length = word_length

    def get_vocabulary(self):
        return self.vocabulary

    def name_object(self, context: list[int], word: str):
        has_name = False
        for obj in context:
            if obj in self.vocabulary:
                for obj_name in self.vocabulary[obj]:
                    if obj_name == word:
                        self.vocabulary[obj][word] += 1
                        # print(self.vocabulary[obj][word])
                        has_name = True
        if not has_name:
            self.choose_object_to_name(context, word)

    def choose_object_to_name(self, context: list[int], word: str):
        obj: int = np.random.choice(context)
        if obj in self.vocabulary:
            self.vocabulary[obj][word] = 1
        else:
            self.vocabulary[obj] = {word: 1}

    def generate_name(self, length: int):
        return "".join(random.choices(string.ascii_lowercase, k=length))

    def choose_name_from_context(self, context: list[int]):
        obj = np.random.choice(context)
        if obj in self.vocabulary:
            return reduce(max_word, self.vocabulary[obj].items())[0]
        return self.generate_name(self.word_length)


class Game:
    def __init__(self, num_players, num_objects, max_context_size=3) -> None:
        self.num_players = num_players
        self.num_objects = num_objects
        self.max_context_size = max_context_size
        self.players = [Player() for _ in range(self.num_players)]
        self.objects = list(range(num_objects))

        self.stats = {"stability": [], "mean_word_per_object": []}

    def step(self):
        context = self.generate_context()
        speaker, hearer = self.choose_players()

        speaker_obj_name = speaker.choose_name_from_context(
            context
        )  # wybiera obiekt i zwraca słowo
        hearer.name_object(context, speaker_obj_name)

    def play(self, max_iters):
        iter_range = list(range(max_iters))
        for _ in iter_range:
            self.step()
            self.vocabulary_stability()
            self.mean_word_per_object()

        plt.plot(iter_range, self.stats["stability"])
        plt.title("Stability of the vocab")
        plt.savefig("stability.png")
        plt.cla()
        plt.plot(iter_range, self.stats["mean_word_per_object"])
        plt.title("Mean word per object")
        plt.savefig("mean_word.png")

    def generate_context(self):
        return random.choices(
            self.objects, k=np.random.randint(1, self.max_context_size)
        )

    def choose_players(self):
        speaker, hearer = random.choices(self.players, k=2)
        return speaker, hearer

    def vocabulary_stability(self):
        stability = 0
        for player in self.players:
            stability += self.player_vocab_stabilty(player)

        self.stats["stability"].append(stability / self.num_players)

    def mean_word_per_object(self):
        mean_word = 0
        for player in self.players:
            per_player = 0
            full_vocab = player.get_vocabulary().values()
            for vocab in full_vocab:
                per_player += len(vocab)

            if len(full_vocab):
                mean_word += per_player / len(full_vocab)

        self.stats["mean_word_per_object"].append(mean_word / self.num_players)

    def player_vocab_stabilty(self, player: Player):
        stability = 0
        vocab = player.get_vocabulary()
        for obj in vocab:
            total = reduce(sum_count, vocab[obj].items())[1]
            highiest = reduce(max_word, vocab[obj].items())[1]

            if total:
                stability += highiest / total

        if len(vocab):
            return stability / len(vocab)
        return 0
