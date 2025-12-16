from base_game import BaseGame
import torch


class FuzzyObject(BaseGame):
    def __init__(self) -> None:
        super().__init__()

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

            # fuzzy object

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
