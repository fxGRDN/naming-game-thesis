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
        vocab_size=2**8,  # 2 bytes signed, change to uint if using cuda
        context_size=(2, 3),
        device=torch.device("cpu"),
    ) -> None:

        self.agents = agents
        self.objects = objects
        self.memory = memory
        self.vocab_size = vocab_size
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

    def get_words_from_speakers(self, speakers, contexts) -> tuple[torch.Tensor, torch.Tensor]:

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


        counts = self.state[
            self.instance_ids, speakers, objects_from_context, :, 1
        ]
        max_counts = counts.max(-1, keepdim=True).values
        is_max = (counts == max_counts).float()
        memory_idx = torch.multinomial(is_max, num_samples=1).squeeze(-1)

        return (
            self.communication_channel(self.state[
                self.instance_ids, speakers, objects_from_context, memory_idx, 0
            ]),
            objects_from_context,
        )
    
    def communication_channel(self, words):
        # identity channel
        return words
    

    
    def perception_channel(self, flat_counts: torch.Tensor) -> torch.Tensor:
        """
        Identity function in base game. Override in subclass to implement
        perception obstruction (e.g., remove best match with probability p).
        """
        return flat_counts

     
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

        best_object_idx, best_memory_idx, found_per_hearer = self.find_best_object(
            hearers, words, contexts
        )

        objects_from_context = self.choose_object_from_context(contexts)
        objects_from_context[found_per_hearer] = best_object_idx[found_per_hearer]
        
        self.successful_communications = (topics == objects_from_context).to(
        torch.float32
    )


        # 2. and 3.
        if (~found_per_hearer).any():

            avaible_memory_slot = (
                self.state[self.instance_ids, hearers, objects_from_context, :, 1] == 0
            )  # (n_missing_hearers,)
            set_word_mask = (~found_per_hearer) & avaible_memory_slot.any(dim=-1)

            # has avaible memory slot
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
            # no avaible memory slot, replace weakest
            replace_word_mask = (~found_per_hearer) & (~avaible_memory_slot.any(dim=-1))
            if replace_word_mask.any():
                hearer_counts = self.state[
                    self.instance_ids,
                    hearers,
                    objects_from_context,
                    :,
                    1,
                ]  # (n_missing_hearers, memory)
                min_counts = hearer_counts.min(-1, keepdim=True).values
                is_min = (hearer_counts == min_counts).float()
                memory_idx = torch.multinomial(is_min, num_samples=1).squeeze(-1)

                self.state[
                    self.instance_ids[replace_word_mask],
                    hearers[replace_word_mask],
                    objects_from_context[replace_word_mask],
                    memory_idx[replace_word_mask],
                ] = torch.stack(
                    [words[replace_word_mask], torch.ones_like(words[replace_word_mask])],
                    dim=-1,
                )

        # 1. reinforce
        if found_per_hearer.any():
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

  # ...existing code...
    def find_best_object(
        self, hearers, words, contexts, apply_obstruction: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given hearers, words and contexts, find best object in context for each hearer.
        If apply_obstruction=True, the best match may be removed with some probability,
        forcing selection of the next-best weighted match.
        """
        # all words in hearers' memory
        hearer_words = self.state[self.instance_ids, hearers, :, :, 0]

        # take words only for objects in context
        hearer_words_in_context = hearer_words * (contexts > 0).unsqueeze(-1)

        # positions where word is matched
        matches = (
            hearer_words_in_context == words[:, None, None]
        )  # (n_hearers, objects, memory) bool

        # positive matches per hearer
        hearer_counts = self.state[
            self.instance_ids, hearers, :, :, 1
        ]  # (n_hearers, objects, memory)

        # mask counts where words match
        match_counts = hearer_counts * matches  # (n_hearers, objects, memory)

        # flatten (objects, memory) -> find best (object, memory) combo per hearer
        flat_counts = match_counts.view(match_counts.size(0), -1).float()

        # apply obstruction 
        flat_counts = self.perception_channel(flat_counts)

        # take the max count(s)
        max_flat_counts = flat_counts.max(dim=-1, keepdim=True).values
        # check if any matches were found
        is_max_flat = (flat_counts == max_flat_counts).float()

        # choose one of the best matches randomly
        best_flat_idx = torch.multinomial(is_max_flat, num_samples=1).squeeze(-1)

        # convert flat index back to (object_idx, memory_idx)
        best_object_idx = best_flat_idx // self.memory
        best_memory_idx = best_flat_idx % self.memory

        found_per_hearer = max_flat_counts.squeeze(-1) > 0

        return best_object_idx, best_memory_idx, found_per_hearer


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

        coherence = ((tensor_agents - count_diffs - count_zero) / tensor_agents).mean(-1)

        return coherence
    

    def global_reference_entropy(self) -> torch.Tensor:

        G, N, M = self.state.shape[0], self.agents, self.objects
        V = self.vocab_size

        best_idx = self.state[..., 1].argmax(dim=-1, keepdim=True)
        top_words = self.state[..., 0].gather(-1, best_idx).squeeze(-1).long()

        offsets = (torch.arange(G * M, device=self.device) * V).view(G, 1, M)
        
        counts = torch.bincount(
            (top_words + offsets).view(-1), 
            minlength=G * M * V
        ).view(G, M, V).float()

        word_total_usage = counts.sum(dim=1, keepdim=True)
        
        p_obj_given_word = counts / (word_total_usage + 1e-10)
        
        word_entropies = -torch.sum(
            p_obj_given_word * torch.log2(p_obj_given_word + 1e-10), 
            dim=1
        )
        active_mask = (word_total_usage.squeeze(1) > 0).float()
        mean_ref_entropy = (word_entropies * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1.0)

        return mean_ref_entropy

    def vocab_usage(self):

        usage = (self.state[:, :, :, :, 0] > 0).float().sum(-1) # all used words per object

        return usage.mean((1, 2))


    def step(self, i: int) -> None:
        speakers, hearers = self.choose_agents()

        contexts = self.generate_context()

        words, topics = self.get_words_from_speakers(speakers, contexts)

        self.set_words_for_hearers(hearers, contexts, words, topics)

        self.prune_memory()

    def play(
            self, rounds: int = 1, 
            tqdm_desc: str = "Playing rounds", 
            disable_tqdm: bool = False, 
            tqdm_position: int = 1,
            snapshot_state=None, 
            snap_name = "",
            calc_stats: bool = True
             ) -> None:

        self.stats = torch.zeros((4, rounds, self.game_instances), dtype=torch.float32)

        progress = tqdm.tqdm(range(rounds), desc=tqdm_desc, position=tqdm_position, disable=disable_tqdm)
        for i in progress:
            self.step(i)

            if snapshot_state is not None and i in snapshot_state:
                state_copy = self.state.clone().cpu().numpy()
                np.save(f"data/{snap_name}_state_step_{i}.npy", state_copy)
            if calc_stats:
                self.stats[0, i] = self.success_rate()
                self.stats[1, i] = self.coherence()
                self.stats[2, i] = self.vocab_usage().float()
                self.stats[3, i] = self.global_reference_entropy().float()

            progress.set_postfix(
                {
                    "Vocab Stability": f"{self.stats[1, i].mean().item():.3f}",
                    "memory_usage": f"{torch.cuda.memory_stats(device=self.device)['allocated_bytes.all.current']/(1024**2):.2f} MB",
                }
            )
