import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np


class BaseGame:
    def __init__(
        self, game_instances=1, agents=100, objects=100, device=torch.device("cpu")
    ) -> None:

        self.game_instances = game_instances
        self.agents = agents
        self.objects = objects
        self.batch_size = torch.arange(game_instances)
        self.device = device

        self.state = torch.zeros(
            (game_instances, agents, objects, agents),
            dtype=torch.float32,
            device=self.device,
        )

    def choose_agents(self):
        perms = torch.randperm(self.agents, device=self.device)
        hearers = perms[: self.agents // 2]
        speakers = perms[self.agents // 2 :]
        return speakers, hearers

    def generate_context(self, max_size: int = 3) -> torch.Tensor:
        device = self.device
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

        has_word_for_object = (
            self.state[0, speakers, objects_from_context].sum(dim=-1) > 0
        )

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
        """
        Listener can:
            1. know heard word and see named object in context -> reinforce association
            2. know heard word but not see named object in context -> choose object from context
            3. not know heard word -> choose object from context and associate with word

        """

        word_object_map = self.state.transpose(-1, -2)

        has_object_for_word = (
            word_object_map[0, listeners, words].sum(dim=-1) > 0
        )  # shape: (n_listeners,)

        context_mask = context > 0

        object_from_context = word_object_map[0, listeners, words] * context_mask

        named_object_from_context = (object_from_context > 0).any(dim=-1)

        has_named_object = has_object_for_word & named_object_from_context

        named_not_in_context = has_object_for_word & (~named_object_from_context)

        if (~has_object_for_word).any():
            objects_from_context = self.choose_object_from_context(context)

            self.state[
                0,
                listeners[~has_object_for_word],
                objects_from_context[~has_object_for_word],
                words[~has_object_for_word],
            ] += 1

        if named_not_in_context.any():
            objects_from_context = self.choose_object_from_context(context)

            self.state[
                0,
                listeners[named_not_in_context],
                objects_from_context[named_not_in_context],
                words[named_not_in_context],
            ] += 1

        if (has_named_object).any():

            best_object_for_word = object_from_context.argmax(dim=-1)

            self.state[
                0,
                listeners[has_named_object],
                best_object_for_word[has_named_object],
                words[has_named_object],
            ] += 1

        everything_covered = (
            ~has_object_for_word | named_not_in_context | has_named_object
        ).all()
        exclusive = (
            ~has_object_for_word + named_not_in_context + has_named_object
        ).sum() == len(listeners)
        assert everything_covered and exclusive

    def step(self):
        # losuj agentow
        speakers, listeners = self.choose_agents()
        # generuj kontekst
        context = self.generate_context()
        words = self.get_words_from_speakers(context, speakers)
        self.set_words_for_listeners(context, listeners, words)

    def vocab_entropy(self) -> torch.Tensor:

        # shape: (game_instance, objects, agents)
        best_word_per_agent = self.state.argmax(-1).transpose(-1, -2)

        one_hot = torch.nn.functional.one_hot(
            best_word_per_agent, num_classes=self.agents
        ).float()
        probs = one_hot.transpose(-1, -2).mean(dim=-1)

        entropy = -torch.sum(
            probs * torch.log(probs + 1e-10), dim=-1
        )  # shape: (game_instance,)

        return entropy.mean(dim=-1)

    def vocab_stability(self) -> torch.Tensor:
        # measures global vocabulary stability

        # shape: (game_instance, objects, agents)
        best_word_per_agent = self.state.max(-1)

        val, _ = best_word_per_agent

        stability = val / self.state.sum(dim=-1).clamp(min=1e-10)
        return stability.mean()

    def vocab_uniqueness(self) -> torch.Tensor:

        # shape: (game_instance, objects, agents)
        unique_words = (self.state > 0).any(dim=-2).sum()

        return unique_words

    def play(self, steps=100):
        self.stats = torch.zeros((3, steps), dtype=torch.float32)
        progress = tqdm.tqdm(range(steps), desc="Playing", unit="step")
        for i in progress:
            self.step()
            self.stats[0, i] = self.vocab_entropy()
            self.stats[1, i] = self.vocab_stability()
            self.stats[2, i] = self.vocab_uniqueness()

            progress.set_postfix(
                {
                    "E": f"{self.stats[0, i].item():.4f} \n",
                    "S": f"{self.stats[1, i].item():.4f} \n",
                    "U": f"{self.stats[2, i].item():.4f} \n",
                    "memory_usage": f"{torch.cuda.memory_stats(device=self.device)['allocated_bytes.all.current']/8192} MB",
                }
            )

    def plot_stats(self):
        x_range = np.arange(self.stats.size(1))
        plt.plot(x_range, self.stats[0].cpu().numpy(), label="Vocab Entropy")
        plt.plot(x_range, self.stats[1].cpu().numpy(), label="Vocab Stability")
        plt.plot(x_range, self.stats[2].cpu().numpy(), label="Vocab Uniqueness")
        plt.xlabel("Steps")
        plt.legend()
        plt.title("Vocabulary Entropy over Time")
        plt.savefig("vocab_entropy.png")
