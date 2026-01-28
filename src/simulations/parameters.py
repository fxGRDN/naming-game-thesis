from enum import Enum
import numpy as np

class DefaultParams(Enum):
    POPULATION_SIZE = 16
    OBJECTS_SIZE = 16
    MEMORY_SIZE = 8
    CONTEXT_SIZE = (2, 3)
    VOCAB_SIZE = 2**8


metrics_limits = [
    (None, 1),  # SUCCESS_RATE
    (0, 1),  # CONSENSUS
    (0, DefaultParams.OBJECTS_SIZE.value),  # DICTIONARY_SIZE
    (0, np.log2(DefaultParams.OBJECTS_SIZE.value))  # REFERENCE_ENTROPY
]