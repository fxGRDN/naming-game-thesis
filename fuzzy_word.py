from base_game import BaseGame
import torch


class FuzzyObject(BaseGame):
    def __init__(self) -> None:
        super().__init__()

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
