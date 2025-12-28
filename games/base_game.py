import torch
import matplotlib.pyplot as plt
import tqdm
import numpy as np


class BaseGame:
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**15,  # 2 bytes signed, change to uint if using cuda
        prune_step=50,
        context_size=(1, 3),
        device=torch.device("cpu"),
    ) -> None:

        self.agents = agents
        self.objects = objects
        self.memory = memory
        self.vocab_size = vocab_size
        self.prune_step = prune_step
        self.game_instances = game_instances
        self.context_size = context_size
        self.device = device
        self.fig_prefix = "base_game"

        self.successful_communications = torch.zeros(
            self.game_instances, device=self.device
        )

        # n_agents, n_objects, memory that holds: (word_id, count)
        self.state = torch.zeros(
            (game_instances, agents, objects, memory, 2),
            dtype=torch.int16,
            device=self.device,
        )

        if self.context_size[1] > self.objects:
            raise ValueError(
                f"Max context size {self.context_size[1]} cannot be larger than number of objects {self.objects}"
            )

        self.instance_ids = torch.arange(game_instances, device=self.device)

    def choose_agents(self):
        scores = torch.rand((self.game_instances, self.agents), device=self.device)
        pairs = torch.topk(scores, 2, dim=1).indices  # Shape: (num_instances, 2)
        speakers = pairs[:, 0]  # (num_instances,)
        hearers = pairs[:, 1]  # (num_instances,)

        return speakers, hearers

    def generate_context(self) -> torch.Tensor:
        n = self.game_instances
        o = self.objects
        max_k = self.context_size[1]

        # how many objects in context per speaker
        ks = torch.randint(self.context_size[0], max_k + 1, (n,), device=self.device)

        probs = torch.ones((n, o), device=self.device, dtype=torch.float32)

        # sample all objects with equal probability
        samples = torch.multinomial(probs, num_samples=max_k, replacement=False)

        idxs = torch.arange(max_k, device=self.device).unsqueeze(0)

        keep_mask = idxs < ks.unsqueeze(1)

        rows_flat = samples[keep_mask]

        cols = torch.arange(n, device=self.device).unsqueeze(1).expand(-1, max_k)
        cols_flat = cols[keep_mask]

        context = torch.zeros((o, n), device=self.device, dtype=torch.float32)

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
        chosen = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
        return chosen

    def gen_words(self, n) -> torch.Tensor:
        return torch.randint(
            1, self.vocab_size, (n,), device=self.device, dtype=torch.int16
        )

    def get_words_from_speakers(self, speakers, contexts) -> torch.Tensor:

        objects_from_context = self.choose_object_from_context(contexts)

        has_name_for_object = (
            self.state[self.instance_ids, speakers, objects_from_context, :, 1].sum(-1)
            > 0
        )

        if (~has_name_for_object).any():
            n_words = (~has_name_for_object).sum().item()
            words = self.gen_words(n_words)

            idx_i = self.instance_ids[~has_name_for_object]
            idx_s = speakers[~has_name_for_object]
            idx_o = objects_from_context[~has_name_for_object]

            self.state[idx_i, idx_s, idx_o, 0, 0] = words
            self.state[idx_i, idx_s, idx_o, 0, 1] = 1

        memory_idx = self.state[
            self.instance_ids, speakers, objects_from_context, :, 1
        ].argmax(-1)

        return (
            self.state[
                self.instance_ids, speakers, objects_from_context, memory_idx, 0
            ],
            objects_from_context,
        )

    def set_words_for_hearers(self, hearers, contexts, words, topics) -> None:
        """
        Hearer can:
            1. know heard word and see named object in context -> reinforce association
            2. know heard word but not see named object in context -> choose object from context
            3. not know heard word -> choose object from context and associate with word

        """

        self.successful_communications = torch.zeros_like(
            self.successful_communications
        )

        objects_from_context = self.choose_object_from_context(contexts)

        # 1. check if memory holds the word of objects in context
        hearer_words = self.state[self.instance_ids, hearers, :, :, 0]

        # mask words not in context
        hearer_words_in_context = hearer_words * (contexts > 0).unsqueeze(-1)

        # words: (n_hearers,)
        # hearer_words_in_context: (n_hearers, objects, memory)

        matches = (
            hearer_words_in_context == words[:, None, None]
        )  # (n_hearers, objects, memory) bool
        found_per_hearer = matches.any(dim=(1, 2))  # (n_hearers,) bool

        # 2. and 3.
        if (~found_per_hearer).any():

            avaible_memory_slot = (
                self.state[self.instance_ids, hearers, objects_from_context, :, 1] == 0
            )  # (n_missing_hearers,)
            set_word_mask = (~found_per_hearer) & avaible_memory_slot.any(dim=-1)

            if set_word_mask.any():
                first_empty_idx = avaible_memory_slot.float().argmax(dim=-1)

                self.state[
                    self.instance_ids[set_word_mask],
                    hearers[set_word_mask],
                    objects_from_context[set_word_mask],
                    first_empty_idx[set_word_mask],
                ] = torch.stack(
                    [words[set_word_mask], torch.ones_like(words[set_word_mask])],
                    dim=-1,
                )

        # 1. find memory idx and reinforce
        if found_per_hearer.any():

            # Get counts for matching positions, zero elsewhere
            hearer_counts = self.state[
                self.instance_ids, hearers, :, :, 1
            ]  # (n_hearers, objects, memory)
            match_counts = hearer_counts * matches  # (n_hearers, objects, memory)

            # Flatten (objects, memory) -> find best (object, memory) combo per hearer
            flat_counts = match_counts.view(
                match_counts.size(0), -1
            )  # (n_hearers, objects*memory)
            best_flat_idx = flat_counts.argmax(dim=-1)  # (n_hearers,)

            # Convert flat index back to (object_idx, memory_idx)
            best_object_idx = best_flat_idx // self.memory
            best_memory_idx = best_flat_idx % self.memory

            self.successful_communications = (topics == best_object_idx).to(
                torch.float32
            )

            # Damp everything by -1 (clamped at 0)
            self.state[
                self.instance_ids[found_per_hearer],
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                :,
                1,
            ] = torch.clamp(
                self.state[
                    self.instance_ids[found_per_hearer],
                    hearers[found_per_hearer],
                    best_object_idx[found_per_hearer],
                    :,
                    1,
                ]
                - 1,
                min=0,
            )

            # Reinforce: increment the best match
            self.state[
                self.instance_ids[found_per_hearer],
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                best_memory_idx[found_per_hearer],
                1,
            ] = torch.clamp(
                self.state[
                    self.instance_ids[found_per_hearer],
                    hearers[found_per_hearer],
                    best_object_idx[found_per_hearer],
                    best_memory_idx[found_per_hearer],
                    1,
                ]
                + 2,
                max=100,
            )

    def prune_memory(self):

        non_zero = self.state[:, :, :, :, 1] != 0

        self.state *= non_zero.unsqueeze(-1)

    def success_rate(self) -> torch.Tensor:
        return self.successful_communications

    def coherence(self) -> torch.Tensor:
        tensor_agents = torch.tensor(self.agents, device=self.device)
        top_word_index = self.state[:, :, :, :, 1].argmax(
            dim=-1
        )  # (games, agents, objects)

        top_words = self.state[
            self.instance_ids.unsqueeze(1).unsqueeze(2),
            torch.arange(self.agents, device=self.device).unsqueeze(1),
            torch.arange(self.objects, device=self.device).unsqueeze(0),
            top_word_index,
            0,
        ].transpose(
            -1, -2
        )  # (games, objects, agents)

        sorted_words, _ = torch.sort(top_words, dim=2)

        count_zero = (sorted_words == 0).sum(dim=2)  # (objects, 1)

        diff = torch.diff(sorted_words, dim=2)  # (agents-1, objects)

        count_diffs = (diff != 0).sum(dim=2)  # (agents-1,)

        coherence = ((tensor_agents - count_diffs - count_zero) / tensor_agents).mean()

        return coherence
    

    def count_polysems(self) -> torch.Tensor:
        tensor_objects_count = torch.tensor(self.objects, device=self.device)

        top_word_index = self.state[:, :, :, :, 1].argmax(dim=-1)  # (agents, objects)
        top_words = self.state[
            self.instance_ids.unsqueeze(1).unsqueeze(2),
            torch.arange(self.agents, device=self.device).unsqueeze(1),
            torch.arange(self.objects, device=self.device).unsqueeze(0),
            top_word_index,
            0,
        ].transpose(
            -1, -2
        )  # (objects, agents)

        top_for_object = top_words.mode(dim=2).values

        sorted_words, _ = torch.sort(top_for_object, dim=1)

        diff = torch.diff(sorted_words, dim=1)  # (agents-1, objects)

        count_diffs = (diff != 0).sum(dim=1) + 1  # (agents-1,)
        return tensor_objects_count / count_diffs



        print(uniqe_words_counts)

        return tensor_objects_count / uniqe_words_counts


    def vocab_usage(self):

        usage = (self.state[:, :, :, :, 0] > 0).float().sum(-1)

        return usage.mean()


    def step(self, i: int) -> None:
        speakers, hearers = self.choose_agents()

        contexts = self.generate_context()

        words, topics = self.get_words_from_speakers(speakers, contexts)

        self.set_words_for_hearers(hearers, contexts, words, topics)

        if (i + 1) % self.prune_step == 0:
            self.prune_memory()

    def play(self, rounds: int = 1) -> None:

        self.stats = torch.zeros((4, rounds, self.game_instances), dtype=torch.float32)

        progress = tqdm.tqdm(range(rounds), desc="Playing rounds", position=3)
        for i in progress:
            self.step(i)

            self.stats[0, i] = self.success_rate()
            self.stats[1, i] = self.coherence()
            self.stats[2, i] = self.vocab_usage().float()
            self.stats[3, i] = self.count_polysems().float()
            # # --- IGNORE ---

            # progress.set_postfix(
            #     {
            #         "Vocab Stability": f"{self.stats[1, i].item():.3f}",
            #         "memory_usage": f"{torch.cuda.memory_stats(device=self.device)['allocated_bytes.all.current']/8192} MB",
            #     }
            # )

    def plot_stats(self) -> None:
        rounds = self.stats.shape[1]
        x = torch.arange(rounds).cpu().numpy()

        def mean_q(metric_idx):
            data = self.stats[metric_idx].cpu().numpy()  # shape (steps, iters)
            mean = data.mean(axis=-1)
            lo = np.percentile(data, 0.5, axis=1)
            hi = np.percentile(data, 99.5, axis=1)
            return mean, lo, hi

        plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        mean, lo, hi = mean_q(0)
        plt.plot(x, mean, color="C0")
        plt.hlines(1.0, 0, rounds, colors="gray", linestyles="dashed", alpha=0.5)
        plt.title("Success Rate")
        plt.xlabel("Rounds")
        plt.ylim(0, 1.2)

        plt.subplot(2, 2, 2)
        mean, lo, hi = mean_q(1)
        plt.plot(x, mean, color="C1")
        plt.hlines(1.0, 0, rounds, colors="gray", linestyles="dashed", alpha=0.5)
        plt.fill_between(x, lo, hi, color="C1", alpha=0.2)
        plt.title("Coherence")
        plt.xlabel("Rounds")
        plt.ylim(0, 1.2)

        plt.subplot(2, 2, 3)
        mean, lo, hi = mean_q(2)
        plt.plot(x, mean, color="C2")
        plt.hlines(1.0, 0, rounds, colors="gray", linestyles="dashed", alpha=0.5)
        plt.fill_between(x, lo, hi, color="C2", alpha=0.2)
        plt.title("Vocab Usage")
        plt.xlabel("Rounds")

        plt.subplot(2, 2, 4)
        mean, lo, hi = mean_q(3)
        plt.fill_between(x, lo, hi, color="C3", alpha=0.2)
        plt.hlines(1.0, 0, rounds, colors="gray", linestyles="dashed", alpha=0.5)
        plt.plot(x, mean, color="C3")
        plt.title("Word per Object Ratio")
        plt.xlabel("Rounds")

        plt.savefig(f"plots/{self.fig_prefix}_stats.png")
