import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import get_default_device
from skimage import measure

sns.set_theme(style="whitegrid")

def gen_word_object(sampling_freq: int = 100):
    device = get_default_device()

    obj_conf = torch.linspace(0, 1, 30, device=device)
    bit_flip_prob = torch.linspace(0, 1, 30, device=device)

    os.makedirs(f'plots/word_object_game', exist_ok=True)

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    STATS_2D = np.zeros((4, 4, len(obj_conf), len(bit_flip_prob)))

    CONSENSUS_MEAN_TIME = np.zeros((len(obj_conf), len(bit_flip_prob)))
    
    data_stat_samples = 0

    try:
        actual_p = []
        for i, obj_conf_value in enumerate(obj_conf):
            actual_p.append(obj_conf_value.item()) 
            data = torch.from_numpy(np.load(f'data/word_object_game/monte_carlo_stats_part_{i}.npy'))
            for j, p in enumerate(bit_flip_prob):
                data_slice = data[j].to(device)

                data_stat_samples = data_slice.shape[1]
                for l, k in enumerate([99, 299, 499, 999]):
                    STATS_2D[0, l, i, j] = data_slice[0, k].mean()
                    STATS_2D[1, l, i, j] = data_slice[1, k].mean()
                    STATS_2D[2, l, i, j] = data_slice[2, k].mean()
                    STATS_2D[3, l, i, j] = data_slice[3, k].mean()

                mean_consensus_time = (data_slice[1].T > 0.90).float().argmax(dim=1)
                mask = mean_consensus_time != 0
                CONSENSUS_MEAN_TIME[i, j] = mean_consensus_time[mask].float().mean().item()*sampling_freq

            print(f'Finished obj confusion {i+1}/{len(obj_conf)}')
            del data
    except FileNotFoundError:
        pass

    if isinstance(bit_flip_prob, torch.Tensor):
        bit_flip_prob = bit_flip_prob.cpu().numpy()
    if isinstance(obj_conf, torch.Tensor):
        obj_conf = obj_conf.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([99, 299, 499, 999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            STATS_2D[0, l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Sukces interakcji'},
            vmax=1,
            square=True,
            ax=ax
        )

        cs = ax.contour(
            STATS_2D[0, l].T,
            levels=[0.7, 0.8, 0.9],
            colors='black',
            linewidths=1,
            linestyles='dashed'
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'{x*100:.0f}%')            

        contours = measure.find_contours(STATS_2D[0, l].T, level=0.90)
        if contours:
            longest_contour = max(contours, key=len)
            contour_x = np.interp(longest_contour[:, 1], np.arange(STATS_2D[0, l].T.shape[1]), bit_flip_prob)
            contour_y = np.interp(longest_contour[:, 0], np.arange(STATS_2D[0, l].T.shape[0]), actual_p)
            print(f'Contour points for 90% success at iteration {(k+1)*sampling_freq}:')
            for x_val, y_val in zip(contour_x, contour_y):
                print(f'Word Flip Prob: {x_val:.4f}, Object Confusion Prob: {y_val:.4f}')

        ax.invert_yaxis()
        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')

    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Szansa na 90% sukces interakcji w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/success.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([99, 299, 499, 999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            STATS_2D[1, l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Konsensus'},
            vmax=1,
            square=True,
            ax=ax
        )

        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
        ax.invert_yaxis()
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Szansa na 90% konsensus w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/consensus.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([99, 299, 499, 999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            STATS_2D[2, l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Rozmiar słownika'},
            square=True,
            vmin=0,
            vmax=8,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
    
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Rozmiar słownika w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/dict.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([99, 299, 499, 999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            STATS_2D[3, l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Entropia'},
            square=True,
            vmin=0,
            vmax=4, # max entropy with 16 objects
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Entropia w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/entropy.png')
    plt.close()


    # Option 1: Using booktabs (recommended - cleaner look)
    print("\n% LaTeX table for consensus average steps")
    print("% Requires: \\usepackage{booktabs}")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Średnia liczba kroków do osiągnięcia 90\\% konsensusu}")
    print("\\label{tab:consensus_mean_time}")
    
    cols = [i for i in range(0, len(actual_p), 4)]
    header = "\\begin{tabular}{c" + "c" * len(cols) + "}"
    print(header)
    print("\\toprule")
    print("Słowo / Obiekt & " + " & ".join([f"{actual_p[i]:.2f}" for i in cols]) + " \\\\")
    print("\\midrule")
    
    for j in range(0, len(bit_flip_prob), 4):
        row_values = [f"{CONSENSUS_MEAN_TIME[i, j]:.0f}" if not np.isnan(CONSENSUS_MEAN_TIME[i, j]) else f"{sampling_freq * data_stat_samples}+" for i in cols]
        print(f"{bit_flip_prob[j]:.2f} & " + " & ".join(row_values) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


gen_word_object(sampling_freq=100)
