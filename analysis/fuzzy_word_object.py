import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(style="whitegrid")



def gen_word_object(
    bit_flip_probs,
    object_confusion_probs,
):
    # szansa na osiagegniecie konsensusu
    os.makedirs(f'plots/word_object_game', exist_ok=True)

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    image = np.zeros((len(bit_flip_probs), len(object_confusion_probs)))
    for i, obj_conf in enumerate(object_confusion_probs):
        data = np.load(f'data/word_object_game/monte_carlo_stats_part_{i}.npy')
        for j, bit_flip_prob in enumerate(bit_flip_probs):
            image [i, j] = (data[j, 1].T > 0.99).any(axis=1).mean()
        del data    

    sns.heatmap(
        image,
        xticklabels=[f'{p:.2f}' for p in object_confusion_probs],
        yticklabels=[f'{p:.2f}' for p in bit_flip_probs],
        cmap='viridis',
        cbar_kws={'label': 'Szansa na osiągnięcie konsensusu'},
    )
    plt.xlabel('Szansa Pomylenia Obiektu')
    plt.ylabel('Szansa Pomylenia Słowa')
    plt.title('Szansa na osiągnięcie konsensusu w zależności od szansy pomylenia słowa i obiektu')
    plt.savefig(f'plots/word_object_game/bit_flip_vs_obj_confusion_consensus.png')
    plt.close()



obj_conf = np.linspace(0.01, 0.3, 25)
bit_flip_prob = np.linspace(0.01, 0.3, 25)

gen_word_object(
    bit_flip_probs=bit_flip_prob,
    object_confusion_probs=obj_conf,
)
