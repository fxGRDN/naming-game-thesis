import torch
import tqdm
import matplotlib.pyplot as plt


class BaseGame:
    def __init__(
        self,
        agents=100,
        objects=100,
        memory=5,
        vocab_size=10**6,
        prune_step=100,
        max_agent_pairs=None,
        device=torch.device("cpu"),
    ) -> None:

        self.agents = agents
        self.objects = objects
        self.memory = memory
        self.vocab_size = vocab_size
        self.prune_step = prune_step
        self.unique_words = set()
        self.max_agent_pairs = max_agent_pairs
        self.device = device
        self.successful_communications = 0
        self.fig_prefix = "base_game"

        # n_agents, n_objects, memory that holds: (word_id, count)
        self.state = torch.zeros(
            (agents, objects, memory, 2), dtype=torch.long, device=self.device
        )

        self.n_pairs = self.agents // 2

        if self.max_agent_pairs is not None:
            self.n_pairs = min(self.n_pairs, self.max_agent_pairs)

    def choose_agents(self):
        perms = torch.randperm(self.agents)


            

        hearers = perms[: self.agents // 2][: self.n_pairs]
        speakers = perms[self.agents // 2 :][: self.n_pairs]

        return speakers, hearers

    def generate_context(self, max_size: int = 3) -> torch.Tensor:
        n = self.n_pairs
        o = self.objects
        max_k = max_size

        # how many objects in context per speaker
        ks = torch.randint(1, max_k + 1, (n,), device=self.device)

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
        return torch.randint(1, self.vocab_size, (n,), device=self.device)

    def get_words_from_speakers(self, speakers, contexts) -> torch.Tensor:

        objects_from_context = self.choose_object_from_context(contexts)

        has_name_for_object = (
            self.state[speakers, objects_from_context, :, 1].sum(-1) > 0
        )

        if (~has_name_for_object).any():
            n_words = (~has_name_for_object).sum().item()
            words = self.gen_words(n_words)

            self.unique_words.update(words.tolist())

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

    def set_words_for_hearers(self, hearers, contexts, words) -> None:
        """
        Hearer can:
            1. know heard word and see named object in context -> reinforce association
            2. know heard word but not see named object in context -> choose object from context
            3. not know heard word -> choose object from context and associate with word

        """

        self.successful_communications = 0

        objects_from_context = self.choose_object_from_context(contexts)

        # 1. check if memory holds the word of objects in context
        hearer_words = self.state[hearers, :, :, 0]

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
                self.state[hearers, objects_from_context, :, 1] == 0
            )  # (n_missing_hearers,)
            set_word_mask = (~found_per_hearer) & avaible_memory_slot.any(dim=-1)

            if set_word_mask.any():
                first_empty_idx = avaible_memory_slot.float().argmax(dim=-1)

                self.state[
                    hearers[set_word_mask],
                    objects_from_context[set_word_mask],
                    first_empty_idx[set_word_mask],
                ] = torch.stack(
                    [words[set_word_mask], torch.ones_like(words[set_word_mask])],
                    dim=-1,
                ).long()

        # 1. find memory idx and reinforce
        if found_per_hearer.any():

            self.successful_communications = found_per_hearer.sum().item()
            # two objects could have the same word, so we take one with highest association

            # Get counts for matching positions, zero elsewhere
            hearer_counts = self.state[hearers, :, :, 1]  # (n_hearers, objects, memory)
            match_counts = hearer_counts * matches  # (n_hearers, objects, memory)

            # Flatten (objects, memory) -> find best (object, memory) combo per hearer
            flat_counts = match_counts.view(
                match_counts.size(0), -1
            )  # (n_hearers, objects*memory)
            best_flat_idx = flat_counts.argmax(dim=-1)  # (n_hearers,)

            # Convert flat index back to (object_idx, memory_idx)
            best_object_idx = best_flat_idx // self.memory
            best_memory_idx = best_flat_idx % self.memory

            # Reinforce: increment the count
            self.state[
                hearers[found_per_hearer],
                best_object_idx[found_per_hearer],
                best_memory_idx[found_per_hearer],
                1,
            ] += 1

    def prune_memory(self):
        """Remove competing words, keep only the dominant one per (agent, object)."""
        # For each (agent, object), find memory slot with highest count
        best_memory_idx = self.state[:, :, :, 1].argmax(dim=-1)  # (agents, objects)

        # Create mask: True only for the best slot
        memory_range = torch.arange(self.memory, device=self.device)
        keep_mask = memory_range == best_memory_idx.unsqueeze(
            -1
        )  # (agents, objects, memory)

        # Zero out non-best slots
        self.state = self.state * keep_mask.unsqueeze(-1)

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

        return ((tensor_agents - count_diffs - count_zero)/ tensor_agents).mean()
    
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


        


    def step(self) -> None:
        speakers, hearers = self.choose_agents()

        contexts = self.generate_context()

        words = self.get_words_from_speakers(speakers, contexts)

        self.set_words_for_hearers(hearers, contexts, words)

    def play(self, rounds: int = 1) -> None:

        self.stats = torch.zeros((4, rounds), dtype=torch.float32)

        # progress = tqdm.tqdm(, desc="Playing rounds", position=3)
        for i in range(rounds):
            self.step()

            self.stats[0, i] = self.success_rate()
            self.stats[1, i] = self.coherence()
            self.stats[2, i] = self.unique_words_per_object().float()
            self.stats[3, i] = self.count_polysems().float()
              # --- IGNORE ---

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
