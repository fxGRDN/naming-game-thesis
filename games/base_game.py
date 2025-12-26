import torch
import tqdm
import matplotlib.pyplot as plt


class BaseGame:
    def __init__(
        self,
        game_instances=1,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=2**16,  # 2 bytes
        prune_step=50,
        max_agent_pairs=None,
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
        self.successful_communications = 0
        self.fig_prefix = "base_game"

        # n_agents, n_objects, memory that holds: (word_id, count)
        self.state = torch.zeros(
            (game_instances, agents, objects, memory, 2),
            dtype=torch.uint16,
            device=self.device,
        )

        if self.context_size[1] > self.objects:
            raise ValueError(
                f"Max context size {self.context_size[1]} cannot be larger than number of objects {self.objects}"
            )

        self.weights = torch.ones(self.agents, device=self.device)
        self.instance_ids = torch.arange(game_instances, device=self.device)

    def choose_agents(self):
        """
        Sequential Sampling:
        Selects 1 Speaker and 1 Hearer uniformly at random.
        Constraint: Speaker != Hearer.
        """
        indices = torch.multinomial(self.weights, num_samples=2, replacement=False)

        speaker = indices[0].unsqueeze(0)  # Shape: (1,)
        hearer = indices[1].unsqueeze(0)  # Shape: (1,)

        return speaker, hearer

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
            1, self.vocab_size, (n,), device=self.device, dtype=torch.float16
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

        self.successful_communications = 0

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

        non_zero = self.state[:, :, :, 1] != 0

        self.state *= non_zero.unsqueeze(-1)

    def unique_words_per_object(self) -> torch.Tensor:
        return torch.tensor(len(self.unique_words), device=self.device)

    def success_rate(self) -> torch.Tensor:
        n_communications = self.n_pairs
        if n_communications == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.tensor(
            self.successful_communications / n_communications, device=self.device
        )

    def coherence(self) -> torch.Tensor:
        tensor_agents = torch.tensor(self.agents, device=self.device)
        top_word_index = self.state[:, :, :, 1].argmax(dim=-1)  # (agents, objects)
        top_words = self.state[
            torch.arange(self.agents, device=self.device).unsqueeze(1),
            torch.arange(self.objects, device=self.device).unsqueeze(0),
            top_word_index,
            0,
        ].t()  # (objects, agents)

        sorted_words, _ = torch.sort(top_words, dim=1)

        count_zero = (sorted_words == 0).sum(dim=1, keepdim=True)  # (objects, 1)

        diff = torch.diff(sorted_words, dim=1)  # (agents-1, objects)
        count_diffs = (diff != 0).sum(dim=1)  # (agents-1,)

        return ((tensor_agents - count_diffs - count_zero) / tensor_agents).mean()

    def count_polysems(self) -> torch.Tensor:
        tensor_objects = torch.tensor(self.objects, device=self.device)

        top_word_index = self.state[:, :, :, 1].argmax(dim=-1)  # (agents, objects)
        top_words = self.state[
            torch.arange(self.agents, device=self.device).unsqueeze(1),
            torch.arange(self.objects, device=self.device).unsqueeze(0),
            top_word_index,
            0,
        ].t()

        top_for_object = top_words.mode(dim=1).values

        return tensor_objects / torch.unique(top_for_object).shape[0]

    def step(self, i: int) -> None:
        speakers, hearers = self.choose_agents()

        contexts = self.generate_context()

        words, topics = self.get_words_from_speakers(speakers, contexts)

        self.set_words_for_hearers(hearers, contexts, words, topics)

        if (i + 1) % self.prune_step == 0:
            self.prune_memory()

    def play(self, rounds: int = 1) -> None:

        self.stats = torch.zeros((4, rounds), dtype=torch.float32)

        # progress = tqdm.tqdm(, desc="Playing rounds", position=3)
        for i in range(rounds):
            self.step(i)

            # self.stats[0, i] = self.success_rate()
            # self.stats[1, i] = self.coherence()
            # self.stats[2, i] = self.unique_words_per_object().float()
            # self.stats[3, i] = self.count_polysems().float()
            # # --- IGNORE ---

            # progress.set_postfix(
            #     {
            #         "Vocab Stability": f"{self.stats[1, i].item():.3f}",
            #         "memory_usage": f"{torch.cuda.memory_stats(device=self.device)['allocated_bytes.all.current']/8192} MB",
            #     }
            # )

    def plot_stats(self) -> None:
        rounds = self.stats.size(1)
        x = torch.arange(rounds).cpu().numpy()

        plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        plt.plot(x, self.stats[0].cpu().numpy())
        plt.title("Success Rate")
        plt.xlabel("Rounds")
        plt.ylim(0, 1.2)

        plt.subplot(2, 2, 2)
        plt.plot(x, self.stats[1].cpu().numpy())
        plt.title("Coherence")
        plt.xlabel("Rounds")
        plt.ylim(0, 1.2)

        plt.subplot(2, 2, 3)
        plt.plot(x, self.stats[2].cpu().numpy())
        plt.title("Unique Words")
        plt.xlabel("Rounds")

        plt.subplot(2, 2, 4)
        plt.plot(x, self.stats[3].cpu().numpy())
        plt.title("Word per Object Ratio")
        plt.xlabel("Rounds")

        plt.savefig(f"plots/{self.fig_prefix}_stats.png")
