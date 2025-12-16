import torch
import tqdm
import matplotlib.pyplot as plt


class BaseGame:
    def __init__(
        self, game_instances=1, agents=100, objects=100, memory=5, vocab_size=10**6
    ) -> None:

        self.game_instances = game_instances
        self.agents = agents
        self.objects = objects
        self.memory = memory
        self.vocab_size = vocab_size

        # n_agents, n_objects, memory that holds: (word_id, count)
        self.state = torch.zeros((agents, objects, memory, 2), dtype=torch.long)

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

    def gen_words(self, n) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (n,), device=self.state.device)

    def get_words_from_speakers(self, speakers, contexts) -> torch.Tensor:

        objects_from_context = self.choose_object_from_context(contexts)

        has_name_for_object = (
            self.state[speakers, objects_from_context, :, 1].sum(-1) > 0
        )

        if (~has_name_for_object).any():
            words = self.gen_words((~has_name_for_object).sum().item())

            self.state[
                speakers[~has_name_for_object],
                objects_from_context[~has_name_for_object],
                0,
            ] = torch.stack(
                [words, torch.ones_like(words)],
                dim=-1,
            ).long()

        memory_idx = self.state[speakers, objects_from_context, :, 1].argmax(-1)

        return self.state[speakers, objects_from_context, memory_idx, 0]

    def set_words_in_listeners(self, listeners, contexts, words) -> None:

        # n_listeners, n_objects, memory
        objects_from_context = self.choose_object_from_context(contexts)
        word_object_map = self.state[listeners, :, :, 0].transpose(-1, -2)
        has_named_object = word_object_map.sum(dim=(1, 2)) > 0

        context_mask = contexts > 0

        named_object_in_context = (word_object_map.sum(dim=1) > 0) * context_mask

        print(named_object_in_context)
        # print(named_object_in_context)

        if (~has_named_object).any():
            self.state[
                listeners[~has_named_object],
                objects_from_context[~has_named_object],
                0,
            ] = torch.stack(
                [words[~has_named_object], torch.ones_like(words[~has_named_object])],
                dim=-1,
            ).long()

    def prune_if_sucessful(self):
        pass

    def play(self, rounds: int = 1) -> None:
        for _ in tqdm.tqdm(range(rounds), desc="Playing rounds"):
            speakers, listeners = self.choose_agents()

            contexts = self.generate_context()

            words = self.get_words_from_speakers(speakers, contexts)

            self.set_words_in_listeners(listeners, contexts, words)
