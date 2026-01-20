import numpy as np
import matplotlib.pyplot as plt

def consensus_threshold():
    bit_flip_prob = np.linspace(0, 1, 100)
    y = np.zeros_like(bit_flip_prob)

    try:
        for i, p in enumerate(bit_flip_prob):
            data = np.load(f"data/word_phase/part_{i}.npy")
            reached_consensus = data[1].mean(axis=-1) > 0.99 / data.shape[2]
            y[i] = reached_consensus.mean()
    except FileNotFoundError:
        pass

    plt.scatter(bit_flip_prob[::-1], y[::-1], s=10)
    plt.xlabel("Szansa Pomylenia Słowa")
    plt.ylabel("Szansa na Osiągnięcie Konsensusu")
    plt.title("Próg Konsensusu w Grze Słowo-Obiekt")
    plt.savefig("plots/word_game/consensus_threshold.png")
    plt.close()


consensus_threshold()