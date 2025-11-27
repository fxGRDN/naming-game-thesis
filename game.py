import torch
import tqdm
import matplotlib.pyplot as plt


class BaseGame:
    def __init__(self, game_instances=1, agents=100, objects=100) -> None:

        self.game_instances = game_instances
        self.agents = agents
        self.objects = objects
        self.batch_size = torch.arange(game_instances)

        self.state = torch.zeros(
            (game_instances, agents, objects, agents), dtype=torch.float32
        )

    def choose_agents(self):
        perms = torch.randperm(self.agents)
        return perms[self.agents // 2 :], perms[: self.agents // 2]

    def generate_context(self, max_size: int = 3) -> torch.Tensor:
        device = self.state.device
        n = self.agents // 2
        o = self.objects
        max_k = max_size

        ks = torch.randint(1, max_k + 1, (n,), device=device)

        probs = torch.ones((n, o), device=device, dtype=torch.float32)

        samples = torch.multinomial(probs, num_samples=max_k, replacement=False)

        idxs = torch.arange(max_k, device=device).unsqueeze(0)
        keep_mask = idxs < ks.unsqueeze(1)

        rows_flat = samples[keep_mask]
        cols = torch.arange(n, device=device).unsqueeze(1).expand(-1, max_k)
        cols_flat = cols[keep_mask]

        context = torch.zeros((o, n), device=device, dtype=torch.float32)

        context[rows_flat, cols_flat] = 1.0 / ks[cols_flat].to(torch.float32)

        # return context of shape (n_speakers, objects)
        return context.t()

    def choose_object_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Given a (objects, n_speakers) context matrix, sample one object per
        speaker according to the context probabilities.
        """
        # context: (objects, n_speakers) -> probs: (n_speakers, objects)
        probs = context.contiguous()
        # if any row sums to zero, replace that row with uniform weights
        row_sums = probs.sum(dim=1, keepdim=True)
        if (row_sums == 0).any():
            probs = torch.where(row_sums == 0, torch.ones_like(probs), probs)
        # sample one object per speaker
        chosen = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
        return chosen

    def get_words_from_speakers(
        self, context: torch.Tensor, speakers: torch.Tensor
    ) -> torch.Tensor:
        objects_from_context = self.choose_object_from_context(context)

        has_word_for_object = self.state[0, speakers, objects_from_context].sum(1) > 0

        if (~has_word_for_object).any():
            self.state[
                0,
                speakers[~has_word_for_object],
                objects_from_context[~has_word_for_object],
                speakers[~has_word_for_object],
            ] = 1.0

        words = torch.argmax(self.state[0, speakers, objects_from_context], dim=1)

        return words

    def set_words_for_listeners(
        self, context: torch.Tensor, listeners: torch.Tensor, words: torch.Tensor
    ) -> None:

        obj_from_context = context > 0
        is_object_named = self.state[0, listeners].sum(1) > 0

        named_obj_from_context = obj_from_context & is_object_named

        print(named_obj_from_context)

        objs = self.state[0, listeners]

        print(objs)

        word_object_map = self.state.transpose(-1, -2)
        has_object_for_word = word_object_map[0, listeners, words].sum(1) > 0

        if (~has_object_for_word).any():
            objects_from_context = self.choose_object_from_context(context)

            self.state[
                0,
                listeners[~has_object_for_word],
                objects_from_context[~has_object_for_word],
                words[~has_object_for_word],
            ] = 1.0

        # context shape: (n_listeners, objects)
        # true means object is in context for that listener
        # we want to know which object in listeners vocab is named

    def word_stability(self) -> float:

        xd = torch.argmax(self.state, dim=2)

    def step(self):
        # losuj agentow
        speakers, listeners = self.choose_agents()
        # generuj kontekst
        context = self.generate_context()

        words = self.get_words_from_speakers(context, speakers)

        self.set_words_for_listeners(context, listeners, words)

        self.word_stability()

    def play(self, steps=100):
        for _ in tqdm.tqdm(range(steps)):
            self.step()
