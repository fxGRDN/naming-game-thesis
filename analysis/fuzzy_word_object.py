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

def gen_word_object():
    # szansa na osiagegniecie konsensusu
    obj_conf = torch.linspace(0.01, 0.6, 50, device=device)
    bit_flip_prob = torch.linspace(0.01, 0.3, 25, device=device)
    os.makedirs(f'plots/word_object_game', exist_ok=True)
    plt.figure(figsize=(13, 6))
    plt.tight_layout()
    image = torch.zeros((len(obj_conf), len(bit_flip_prob)), device=device)
    try:
        actual_p = []
        for i, obj_conf_value in enumerate(obj_conf):
            actual_p.append(obj_conf_value.item()) 
            data = torch.from_numpy(np.load(f'data/word_object_game/monte_carlo_stats_part_{i}.npy'))
            for j, p in enumerate(bit_flip_prob):
                data_slice = data[j].to(device)
                val = (data_slice[1].T > 0.99).any(dim=1).float().mean().item()
                image[i, j] = torch.tensor(val, device=device) 
            print(f'Finished obj confusion {i+1}/{len(obj_conf)}')
            print(image[i])
            sns.heatmap(
                image.cpu().numpy().T,
                xticklabels=[f'{p:.2f}' for p in actual_p],
                yticklabels=[f'{p:.2f}' for p in bit_flip_prob],
                cmap='viridis',
                cbar_kws={'label': 'Szansa na osiągnięcie konsensusu'},
            )
            plt.savefig(f'plots/word_object_game/bit_flip_vs_obj_confusion_consensus-{i}.png')
            plt.close()
            del data
    except FileNotFoundError:
        pass

    

    sns.heatmap(
        image.cpu().numpy(),
        xticklabels=[f'{p:.2f}' for p in actual_p],
        yticklabels=[f'{p:.2f}' for p in bit_flip_prob],
        cmap='viridis',
        cbar_kws={'label': 'Szansa na osiągnięcie konsensusu'},
    )
    plt.xlabel('Szansa Pomylenia Obiektu')
    plt.ylabel('Szansa Pomylenia Słowa')
    plt.title('Szansa na osiągnięcie konsensusu w zależności od szansy pomylenia słowa i obiektu')
    plt.savefig(f'plots/word_object_game/bit_flip_vs_obj_confusion_consensus.png')
    plt.close()





gen_word_object()
