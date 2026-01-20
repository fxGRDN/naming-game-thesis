import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

def consensus_threshold():
    bit_flip_prob = np.linspace(0, 1, 100)
    y = np.zeros((len(bit_flip_prob), 4, 100000))
    try:
        for i, p in enumerate(bit_flip_prob):
            data = np.load(f"data/object_phase/part_{i}.npy")
            y[i] = data.mean(axis=-1)
            del data

    except FileNotFoundError:
        pass



    os.makedirs("plots/object_game", exist_ok=True)
    sample_idx = np.arange(y.shape[2])

    Y, X = np.meshgrid(bit_flip_prob, sample_idx)

    stat_names = ['Sukces interakcji', 'Konsensus', 'Rozmiar słownika', 'Entropia referencyjna']

    for stat in range(4):
        Z = y[:, stat, :].T  # shape: (samples, probabilities)

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        axhm = fig.add_subplot(1, 2, 2)
        ax.view_init(elev=25, azim=225)
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)

        ax.invert_xaxis()

        if stat > 1:  # Dictionary Size
            ax.invert_yaxis()
            ax.invert_xaxis()

        ax.set_ylabel("Szansa Pomylenia Obiektu")
        ax.set_xlabel("Indeks Próby")
        ax.set_zlabel("Wartość")

        ax.view_init(elev=35.264, azim=45)  # isometric-like view

        im = axhm.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=(bit_flip_prob.min(), bit_flip_prob.max(), sample_idx.min(), sample_idx.max()),
            cmap="viridis"
        )
        axhm.set_xlabel("Szansa Pomylenia Obiektu")
        axhm.set_ylabel("Indeks Próby")
        fig.colorbar(im, ax=axhm, shrink=0.8)

        fig.suptitle(stat_names[stat])
        plt.tight_layout()
        plt.savefig(f"plots/object_game/consensus_surface_stat_{stat}.png", dpi=200)
        plt.close(fig)


consensus_threshold()