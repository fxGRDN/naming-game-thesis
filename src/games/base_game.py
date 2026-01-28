import torch
import tqdm
import os

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

        # cached top words per (game, agent, object) - computed once per step
        self.top_words = torch.zeros(
            (game_instances, agents, objects),
            dtype=torch.uint8,
            device=self.device,
        )

        # n_agents, n_objects, memory that holds: (word_id, count)
        self.state = torch.zeros(
            (game_instances, agents, objects, memory, 2),
            dtype=torch.uint8,
            device=self.device,
        )

        

        if self.context_size[1] > self.objects:
            raise ValueError(
                f"Max context size {self.context_size[1]} cannot be larger than number of objects {self.objects}"
            )

        self.instance_ids = torch.arange(game_instances, device=self.device)

        # Pre-allocated tensors for generate_context
        self._context_probs = torch.ones((game_instances, objects), device=self.device, dtype=torch.float32)
        self._context_idxs = torch.arange(self.context_size[1], device=self.device).unsqueeze(0)  # (1, max_k)
        self._context_buffer = torch.zeros((game_instances, objects), device=self.device, dtype=torch.float32)
        
        # Pre-allocated tensors for find_best_object
        self._obj_offsets = torch.arange(objects, device=self.device).unsqueeze(0)  # (1, M)
        
        # Pre-allocated tensors for coherence
        self._tensor_agents = torch.tensor(self.agents, device=self.device)
        
        # Pre-allocated tensors for combined metrics (coherence + entropy)
        self._word_counts = torch.zeros((game_instances, objects, vocab_size), device=self.device, dtype=torch.float32)
        self._ones_for_scatter = torch.ones((game_instances, objects, agents), device=self.device, dtype=torch.float32)
        self._coherence = torch.zeros(game_instances, device=self.device, dtype=torch.float32)
        self._entropy = torch.zeros(game_instances, device=self.device, dtype=torch.float32)
        
        # Pre-allocated tensor for ones (used in multiple places)
        self._ones_G = torch.ones(game_instances, dtype=torch.uint8, device=self.device)

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

        # Gumbel-topk trick for sampling without replacement (faster than multinomial)
        gumbel_noise = -torch.log(-torch.log(torch.rand((n, o), device=self.device) + 1e-20) + 1e-20)
        samples = gumbel_noise.topk(max_k, dim=1).indices  # (n, max_k)

        # Create mask for valid samples using pre-allocated idxs
        keep_mask = (self._context_idxs < ks.unsqueeze(1)).float()  # (n, max_k)

        # Build context using scatter_add - zero out and reuse buffer
        self._context_buffer.zero_()
        
        # Probability per object = 1/ks, masked by keep_mask
        probs_per_sample = keep_mask / ks.unsqueeze(1).float()  # (n, max_k)
        
        # Scatter probabilities to context
        self._context_buffer.scatter_add_(1, samples, probs_per_sample)


        return self._context_buffer

    def choose_object_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Given a (n_speakers, objects) context matrix, sample one object per
        speaker according to the context probabilities using Gumbel-max trick.
        """
        # Gumbel-max trick for weighted sampling (faster than multinomial)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(context) + 1e-20) + 1e-20)
        # Add log-probabilities to Gumbel noise, mask out zero-prob with -inf
        log_probs = torch.log(context + 1e-20)
        chosen = (log_probs + gumbel_noise).argmax(dim=-1)


        return chosen

    def gen_words(self, n) -> torch.Tensor:
        return torch.randint(
            1, self.vocab_size, (n,), device=self.device, dtype=torch.uint8
        )

    def get_words_from_speakers(self, speakers, contexts) -> tuple[torch.Tensor, torch.Tensor]:


        # choose objects for everyone
        objects_from_context = self.choose_object_from_context(contexts)

        G, N, M, mem = self.game_instances, self.agents, self.objects, self.memory
        
        # flat indices for (game, speaker, object)
        flat_idx = (self.instance_ids * N * M + speakers * M + objects_from_context)  # (G,)
        
        # flat state for easy indexing
        state_flat = self.state.view(G * N * M, mem, 2)

        # pick memory slots for selected (game, speaker, object)
        selected_slots = state_flat[flat_idx]  # (G, memory, 2)
        

        # word counts
        counts = selected_slots[:, :, 1]  # (G, memory)

        # is object named?
        has_name_for_object = counts.sum(-1) > 0  # (G,)
        
        # generate new words
        new_words = self.gen_words(G)
        
        # mask not named
        no_name_mask = ~has_name_for_object  # (G,)
        
        # update memory slot 0 with new word where no name exists       
        new_word_slot = torch.stack([new_words, self._ones_G], dim=-1)  # (G, 2)

        # update slot 0
        updated_slot0 = torch.where(
            no_name_mask.unsqueeze(-1), # (G, 1)
            new_word_slot, # new word and count=1
            selected_slots[:, 0, :] # existing slot 0
        )  # (G, 2)
        
        # write back updated slot 0
        state_flat[flat_idx, 0] = updated_slot0
        
        # take counts
        counts = state_flat[flat_idx, :, 1].float()  # (G, memory)

        # gumbel-max        
        noise = torch.rand_like(counts) * 0.001
        
        # take best memory slot
        memory_idx = (counts + noise).argmax(dim=-1)  # (G,)
        
        # take word from best slot
        words_out = state_flat[flat_idx, memory_idx, 0]

        # return words and topics
        return (
            self.communication_channel(words_out),
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

        # dims
        G, N, M, mem = self.game_instances, self.agents, self.objects, self.memory


        # find best object in context for each hearer
        best_object_idx, best_memory_idx, found_per_hearer = self.find_best_object(
            hearers, words, contexts
        )

        # choose random object from context for all
        random_objects = self.choose_object_from_context(contexts)
        
        # final chosen object per hearer
        chosen_objects = torch.where(found_per_hearer, best_object_idx, random_objects)
        
        # measure success       
        self.successful_communications = (topics == chosen_objects).to(torch.float32)

        # flat indexes view
        state_flat = self.state.view(G * N * M, mem, 2)
        flat_idx = (self.instance_ids * N * M + hearers * M + chosen_objects).long()  # (G,)
        

        # slots and counts for chosen (game, hearer, object)
        selected_slots = state_flat[flat_idx]  # (G, memory, 2)
        slot_counts = selected_slots[:, :, 1]  # (G, memory)
        
        # empty memory slot
        empty_mask = (slot_counts == 0).float()  # (G, memory)
        has_empty = empty_mask.sum(-1) > 0  # (G,)
        first_empty_idx = empty_mask.argmax(dim=-1)  # (G,)
        
        # gumbel-max
        noise = torch.rand_like(slot_counts.float()) * 0.001
        weakest_idx = (slot_counts.float() + noise).argmin(dim=-1)  # (G,)
        
        # write pointers
        write_slot_idx = torch.where(has_empty, first_empty_idx, weakest_idx)
        
        # build new slot value
        new_slot_value = torch.stack([words, torch.ones_like(words)], dim=-1)  # (G, 2)
        
        # one hot to select slot
        slot_onehot = torch.nn.functional.one_hot(write_slot_idx, num_classes=mem)  # (G, memory)
        
        # update the appropriate slot
        updated_slots = torch.where(
            slot_onehot.unsqueeze(-1).bool(), # is this the write slot?
            new_slot_value.unsqueeze(1).expand(-1, mem, -1), # new slot value
            selected_slots # existing slots
        )  # (G, memory, 2)
        
        # havent found the word-object association, so write new slot
        should_write = ~found_per_hearer  # (G,)
        final_slots = torch.where(
            should_write.unsqueeze(-1).unsqueeze(-1),
            updated_slots, # new slot written
            selected_slots # existing slots
        )  # (G, memory, 2)
        
        # update state
        state_flat[flat_idx] = final_slots
        

        # flat best object
        reinforce_flat_idx = (self.instance_ids * N * M + hearers * M + best_object_idx).long()  # (G,)
        # memory slots to reinforce
        reinforce_slots = state_flat[reinforce_flat_idx]  # (G, memory, 2)
        
        # dampen counts by 1, min 0
        damped_counts = torch.clamp(reinforce_slots[:, :, 1].int() - 1, min=0).to(torch.uint8)  # (G, memory)
        
        # one hot for best memory idx
        boost_onehot = torch.nn.functional.one_hot(best_memory_idx, num_classes=mem)  # (G, memory)

        # boost count by 2 where best memory idx
        boosted_counts = torch.clamp(damped_counts.int() + boost_onehot * 2, max=100).to(torch.uint8)  # (G, memory)
        
  
        # build reinforced slots
        reinforce_updated = reinforce_slots.clone()

        # update counts
        reinforce_updated[:, :, 1] = boosted_counts
        
        # final reinforce slots
        reinforce_final = torch.where(
            found_per_hearer.unsqueeze(-1).unsqueeze(-1),
            reinforce_updated, # boosted counts
            reinforce_slots # existing slots
        )
        
        # write back reinforced slots
        state_flat[reinforce_flat_idx] = reinforce_final

    def find_best_object(
        self, hearers, words, contexts, apply_obstruction: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given hearers, words and contexts, find best object in context for each hearer.
        Uses Gumbel-max trick instead of multinomial for tie-breaking.
        """
        # dims
        G, N, M, mem = self.game_instances, self.agents, self.objects, self.memory
        
        # flatten again
        state_flat = self.state.view(G * N * M, mem, 2)
        
        # flat hearer indices
        hearer_base_idx = (self.instance_ids * N * M + hearers * M).long()  # (G,)

        # gather hearer data
        hearer_indices = hearer_base_idx.unsqueeze(1) + self._obj_offsets  # (G, M) 
        hearer_data = state_flat[hearer_indices.view(-1)].view(G, M, mem, 2)  # (G, M, mem, 2)
        

        # separate words and counts
        hearer_words = hearer_data[:, :, :, 0]  # (G, M, mem)
        hearer_counts = hearer_data[:, :, :, 1]  # (G, M, mem)
        
        # in context mask
        in_context = (contexts > 0).unsqueeze(-1)  # (G, M, 1)
        
        # know + in context matches
        matches = (hearer_words == words[:, None, None]) & in_context  # (G, M, mem)
        
        # mask counts to 0 where no match
        match_counts = hearer_counts * matches  # (G, M, mem)
        
        # flatten
        flat_counts = match_counts.view(G, -1).float()
        
        # pass trough perception channel
        flat_counts = self.perception_channel(flat_counts)
        
        # gumbel-max
        noise = torch.rand_like(flat_counts) * 0.001
        best_flat_idx = (flat_counts + noise).argmax(dim=-1)  # (G,)
        
        # convert flat index back to (object_idx, memory_idx)       
        best_object_idx = best_flat_idx // mem
        best_memory_idx = best_flat_idx % mem
        
        # check if found
        found_per_hearer = flat_counts.max(dim=-1).values > 0

        return best_object_idx.long(), best_memory_idx.long(), found_per_hearer


    def prune_memory(self):
        non_zero = self.state[:, :, :, :, 1] != 0
        self.state.mul_(non_zero.unsqueeze(-1))

    def success_rate(self) -> torch.Tensor:
        return self.successful_communications

    def _compute_top_words(self) -> None:
        """Compute and cache the top word for each (game, agent, object)."""

        # dims
        G, N, M = self.game_instances, self.agents, self.objects
        
        # get counts for all memory slots
        counts = self.state[..., 1].float()  # (G, N, M, memory)
        
        # gumbel-max
        noise = torch.rand_like(counts) * 0.001
        best_memory_idx = (counts + noise).argmax(dim=-1, keepdim=True)  # (G, N, M, 1)
        
        # cache top words for coherence and entropy calculations
        self.top_words = self.state[..., 0].gather(-1, best_memory_idx).squeeze(-1)

    def _compute_metrics(self) -> None:
        """Compute coherence and entropy together, reusing word_counts."""
        G, N, M, V = self.game_instances, self.agents, self.objects, self.vocab_size
        
        # top words per object(games, objects, agents)
        top_words = self.top_words.transpose(-1, -2)
        
        # zero out and reuse pre-allocated word_counts buffer
        self._word_counts.zero_()
        self._word_counts.scatter_add_(2, top_words.long(), self._ones_for_scatter)
        
        # coherence
        max_counts = self._word_counts[:, :, 1:].max(dim=-1).values  # exclude word 0
        self._coherence = (max_counts / N).mean(dim=-1)
        
        # entropy
        word_total_usage = self._word_counts.sum(dim=1, keepdim=True)  # (G, 1, V)
        
        # P(object | word)
        p_obj_given_word = self._word_counts / (word_total_usage + 1e-10)  # (G, M, V)
        
        # entropy per word
        word_entropies = -torch.sum(
            p_obj_given_word * torch.log2(p_obj_given_word + 1e-10),
            dim=1
        )  # (G, V)
        
        # average over active words only
        active_mask = (word_total_usage.squeeze(1) > 0).float()  # (G, V)
        self._entropy = (word_entropies * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp(min=1.0)

    def coherence(self) -> torch.Tensor:
        """Return cached coherence value."""
        return self._coherence
    
    def global_reference_entropy(self) -> torch.Tensor:
        """Return cached entropy value."""
        return self._entropy

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
            sampling_freq: int = 100,
            disable_tqdm: bool = False, 
            tqdm_position: int = 1,
            ma_window: int = 100,
             ) -> None:


        self.step = torch.compile(self.step, mode="default")

        if sampling_freq > 0:
            self.stats = torch.zeros((4, rounds // sampling_freq, self.game_instances), dtype=torch.float32, device=self.device)
            success_buffer = torch.zeros((ma_window, self.game_instances), dtype=torch.float32, device=self.device)
            success_sum = torch.zeros(self.game_instances, dtype=torch.float32, device=self.device)


        progress = tqdm.tqdm(range(rounds), desc=tqdm_desc, position=tqdm_position, disable=disable_tqdm)
        with torch.inference_mode():
            for i in progress:
                self.step(i)
                
                # Update moving average for success rate
                if sampling_freq > 0:
                    buf_idx = i % ma_window
                    old_val = success_buffer[buf_idx].clone()
                    new_val = self.successful_communications.clone()
                    success_buffer[buf_idx] = new_val
                    success_sum = success_sum - old_val + new_val

                if sampling_freq > 0 and i % sampling_freq == 0:
                    self._compute_top_words()
                    self._compute_metrics()
                    # Use moving average for success rate
                    window_len = min(i + 1, ma_window)
                    self.stats[0, i // sampling_freq] = success_sum / window_len
                    self.stats[1, i // sampling_freq] = self.coherence()
                    self.stats[2, i // sampling_freq] = self.vocab_usage().float()
                    self.stats[3, i // sampling_freq] = self.global_reference_entropy().float()
