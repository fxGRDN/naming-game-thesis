import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_theme(style="whitegrid")

def get_default_device()  -> torch.device:
    # prefer CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_default_device(str(device))

    return device


device = get_default_device()

def gen_word_object(ignore_existing: bool = True):

    if not os.path.exists('data/word_object_game/mean/success.npy') or ignore_existing:
        # szansa na osiagegniecie konsensusu
        obj_conf = torch.linspace(0, 0.7, 25, device=device)
        bit_flip_prob = torch.linspace(0, 0.7, 25, device=device)
        os.makedirs(f'plots/word_object_game', exist_ok=True)
        plt.figure(figsize=(13, 6))
        plt.tight_layout()
        SUCESS = np.zeros((4, len(obj_conf), len(bit_flip_prob)))
        CONSENSUS = np.zeros((4, len(obj_conf), len(bit_flip_prob)))
        DICT = np.zeros((4, len(obj_conf), len(bit_flip_prob)))
        ENTROPY = np.zeros((4, len(obj_conf), len(bit_flip_prob)))
        try:
            actual_p = []
            for i, obj_conf_value in enumerate(obj_conf):
                actual_p.append(obj_conf_value.item()) 
                data = torch.from_numpy(np.load(f'data/word_object_game/monte_carlo_stats_part_{i}.npy'))
                for j, p in enumerate(bit_flip_prob):
                    data_slice = data[j].to(device)
                    for l, k in enumerate([9999, 14999, 29999, 49999]):
                        SUCESS[l, i, j] = data_slice[0, : k].float().mean()
                        CONSENSUS[l, i, j] = data_slice[1, : k].float().mean()
                        DICT[l, i, j] = data_slice[2, : k].mean()
                        ENTROPY[l, i, j] = data_slice[3, : k].mean()

                print(f'Finished obj confusion {i+1}/{len(obj_conf)}')
                del data
        except FileNotFoundError:
            pass

        np.save('data/word_object_game/mean/success.npy', SUCESS)
        np.save('data/word_object_game/mean/consensus.npy', CONSENSUS)
        np.save('data/word_object_game/mean/dict.npy', DICT)
        np.save('data/word_object_game/mean/entropy.npy', ENTROPY)

    else:

        SUCESS = np.load('data/word_object_game/mean/success.npy')
        CONSENSUS = np.load('data/word_object_game/mean/consensus.npy')
        DICT = np.load('data/word_object_game/mean/dict.npy')
        ENTROPY = np.load('data/word_object_game/mean/entropy.npy')
        obj_conf = torch.linspace(0, 0.7, 25, device=device)
        bit_flip_prob = torch.linspace(0, 0.7, 25, device=device)
        actual_p = [p.item() for p in obj_conf]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([9999, 14999, 29999, 49999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            SUCESS[l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Sukces interakcji'},
            vmax=1,
            square=True,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {k+1} iteracjach')

    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Szansa na 99% sukces interakcji w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/success.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([9999, 14999, 29999, 49999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            CONSENSUS[l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Konsensus'},
            vmax=1,
            square=True,
            ax=ax
        )

        ax.set_title(f'Po {k+1} iteracjach')
        ax.invert_yaxis()
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Szansna na 99% konsensus w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/consensus.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([9999, 14999, 29999, 49999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            DICT[l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Rozmiar słownika'},
            square=True,
            vmin=0,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {k+1} iteracjach')
    
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Rozmiar słownika w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/dict.png')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for l, k in enumerate([9999, 14999, 29999, 49999]):
        ax = axes[l // 2, l % 2]
        sns.heatmap(
            ENTROPY[l].T,
            xticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(actual_p)],
            yticklabels=[f'{p:.2f}' if i % 4 == 0 else '' for i, p in enumerate(bit_flip_prob)],
            cmap='viridis',
            cbar_kws={'label': 'Entropia'},
            square=True,
            vmin=0,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {k+1} iteracjach')
    
    fig.supxlabel('Szansa Pomylenia Obiektu')
    fig.supylabel('Szansa Pomylenia Słowa')
    plt.suptitle('Entropia w zależności od szansy pomylenia słowa i obiektu', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/entropy.png')
    plt.close()



gen_word_object(ignore_existing=True)
