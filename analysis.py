import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import os
import pandas as pd
import torch
import networkx as nx
import sklearn
sns.set_theme(style="whitegrid")


# data_1_2 = np.load('data/base_game_object_size_consensus.npy')
# data_1_3 = np.load('data/base_game_memory_size_consensus.npy')
# data_1_4 = np.load('data/base_game_vocab_size_consensus.npy')

# mean_1_2 = np.mean(data_1_2, axis=-1)
# mean_1_3 = np.mean(data_1_3, axis=-1)
# mean_1_4 = np.mean(data_1_4, axis=-1)


def success_rate_ma(x, window_size=100, log_plot=True):
    """Calculate moving average of success rate."""
    time_len = x.shape[-2]
    cumsum = np.cumsum(x, axis=-2)
    cumsum_pad = np.concatenate(
        [np.zeros((*x.shape[:-2], 1, x.shape[-1]), dtype=float), cumsum],
        axis=-2,
    )
    

    # indices for window start and end
    start_idx = np.clip(np.arange(time_len) - window_size + 1, 0, time_len)
    end_idx = np.arange(1, time_len + 1)

    # broadcast indices to match x’s shape
    idx_shape = (1,) * (x.ndim - 2) + (time_len, 1)
    start_idx_b = start_idx.reshape(idx_shape)
    end_idx_b = end_idx.reshape(idx_shape)

    start_take = np.take_along_axis(cumsum_pad, start_idx_b, axis=-2)
    end_take = np.take_along_axis(cumsum_pad, end_idx_b, axis=-2)

    window_sum = end_take - start_take
    window_len = np.minimum(np.arange(time_len) + 1, window_size).reshape(idx_shape)

    return window_sum / window_len



def mean_q(data):
    mean = data.mean(axis=-1)
    lo = np.percentile(data, 0.5, axis=1)
    hi = np.percentile(data, 99.5, axis=1)
    return mean, lo, hi




def baseline(path, title_prefix, label_suffix, folder, log_plot=True):
    data_1_0 = np.load(path)
    x = np.arange(data_1_0.shape[-2])
    plt.figure(figsize=(12, 6))

    os.makedirs(f'plots/{folder}', exist_ok=True)

    mean_success_ma = success_rate_ma(data_1_0[0])

    mean, lo, hi = mean_q(mean_success_ma)

    plt.subplot(1, 2, 1)
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title(f'Średni sukces w czasie - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.legend()
    plt.subplot(1, 2, 2)

    time_dist = (mean_success_ma.T > 0.99).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]

    print("time to 99% success:", 
          "\nMean", time_dist.mean(), 
          "\n Std", time_dist.std(), 
          "\n Median", np.median(time_dist),
          "\n Skewness", scipy.stats.skew(time_dist),
          "\n Kurtosis", scipy.stats.kurtosis(time_dist),
          "\n","="*15
          )  
    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 99% sukcesu - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')

    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_success_rate.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    mean, lo, hi = mean_q(data_1_0[1])

    for i in np.random.randint(0, data_1_0[1].shape[1], size=5):
        sns.lineplot(x=x, y=data_1_0[1].T[i], alpha=0.3, color='gray', label='_nolegend_')

    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title(f'Średnia spójność w czasie - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia Spójność')
    plt.legend()
    plt.subplot(1, 2, 2)
    time_dist = (data_1_0[1].T > 0.99).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]

    print("time to 99% consensus:", 
          "\nMean", time_dist.mean(), 
          "\n Std", time_dist.std(), 
          "\n Median", np.median(time_dist),
          "\n Skewness", scipy.stats.skew(time_dist),
          "\n Kurtosis", scipy.stats.kurtosis(time_dist),
          "\n","="*15
          )  



    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 99% spójności - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_consensus.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[2])
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title('Średni rozmiar słownika w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.legend()
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_dict_size.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[3])
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.hlines(0, xmin=0, xmax=len(x)-1, colors='r', linestyles='dashed', label='Brak entropii')
    plt.title(f'Entropia referencyjna - {title_prefix}')
    plt.legend()
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_entropy.png')
    plt.close()



def gen(path, title_prefix, label_suffix, param_values, folder, log_plot=True):
    data = np.load(path) 

    os.makedirs(f'plots/{folder}', exist_ok=True)

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    # mean success rate
    ma_stat_0 = success_rate_ma(data[:, 0])
    for stat, param in zip(ma_stat_0, param_values):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2]), y=stat.mean(axis=-1), label = param)
        plt.subplot(1, 2, 2)
        time_dist = (stat.T > 0.99).argmax(axis=1)
        mask = time_dist != 0
        time_dist = time_dist[mask]

        time_dist_df = pd.DataFrame({label_suffix: [param]*len(time_dist), 'time_to_99_success': time_dist})

        if len(time_dist) > 0:
            sns.boxplot(data=time_dist_df, y='time_to_99_success', x=label_suffix)
    
    plt.subplot(1, 2, 1)
    plt.title(f'Średni sukces w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')

    plt.subplot(1, 2, 2)
    plt.title(f'Czas do osiągnięcia 99% sukcesu vs {title_prefix}')
    plt.ylabel('Kroki Symulacji')
    plt.xlabel(title_prefix)

    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_success_rate.png')
    plt.close()

    ####################################################
    ####################################################
    ####################################################

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    for stat, param in zip(data[:, 1], param_values):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2]), y=stat.mean(axis=-1), label = param)
        plt.subplot(1, 2, 2)
        time_dist = (stat.T > 0.99).argmax(axis=1)
        mask = time_dist != 0
        time_dist = time_dist[mask]
        time_dist_df = pd.DataFrame({label_suffix: [param]*len(time_dist), 'time_to_99_success': time_dist})

        if len(time_dist) > 0:
            sns.boxplot(data=time_dist_df, y='time_to_99_success', x=label_suffix)
    
    plt.subplot(1, 2, 1)
    plt.title(f'Średnia spójność vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')

    plt.subplot(1, 2, 2)
    plt.title(f'Czas do osiągnięcia 99% spójności vs {title_prefix}')
    plt.ylabel('Kroki Symulacji')
    plt.xlabel(title_prefix)

    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_consensus.png')
    plt.close()

    ####################################################
    ####################################################
    ####################################################


    plt.figure(figsize=(13, 6))
    for stat, param in zip(data[:, 2], param_values):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2]), y=stat.mean(axis=-1), label = param)

        plt.subplot(1, 2, 2)
        # czas na osiagniecie peaku
        peaks = stat.T.argmax(axis=1)
        peak_df = pd.DataFrame({label_suffix: [param]*len(peaks), 'time_to_peak_dict_size': peaks})

        sns.boxplot(data=peak_df, y='time_to_peak_dict_size', x=label_suffix)


    plt.subplot(1, 2, 1)
    plt.title(f'Średni rozmiar słownika w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')

    plt.subplot(1, 2, 2)
    plt.title(f'Czas do osiągnięcia maksymalnego rozmiaru słownika vs {title_prefix}')
    plt.ylabel('Kroki Symulacji')
    plt.xlabel(title_prefix)


    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_dict_size.png')
    plt.close()

    ####################################################
    ####################################################
    ####################################################

    if log_plot:
        plt.figure(figsize=(10, 6))
        y = data[:, 2].transpose(0, 2, 1).argmax(axis=2).mean(axis=1)
        plt.loglog(param_values, y, )
        plt.loglog(param_values, np.array(param_values)**(1.5), '--', label='Oczekiwana złożoność $O(n^{1.5})$')
        plt.title(f'Średni czas do osiągnięcia maksymalnego rozmiaru słownika vs {title_prefix}')
        plt.xlabel(title_prefix)
        plt.ylabel('Kroki Symulacji')
        plt.legend()

        plt.savefig(f'plots/{folder}/{label_suffix}_analysis_dict_size_mean_time.png')
        plt.close()

    for stat, param in zip(data[: ,3], param_values):
        sns.lineplot(x=np.arange(stat.shape[-2]), y=stat.mean(axis=-1), label = param)

    plt.title(f'Średnia entropia referencyjna w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia Entropia Referencyjna')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_entropy.png')
    plt.close()





def gen_word_object(
    bit_flip_probs,
    object_confusion_probs,
):
    # szansa na osiagegniecie konsensusu
    data = np.load('data/word_object_game_monte_carlo_stats.npy')

    os.makedirs(f'plots/word_object_game', exist_ok=True)

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    image = np.zeros((len(bit_flip_probs), len(object_confusion_probs)))
    for i, obj_conf in enumerate(object_confusion_probs):
        for j, bit_flip_prob in enumerate(bit_flip_probs):
            stat = data[i, j, 1]
            image [i, j] = (stat.T > 0.99).any(axis=1).mean()


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


            


def power_law():
    data = np.load('data/base_game_population_size_base.npy')
    y = data[:, 2].transpose(0, 2, 1).argmax(axis=2).mean(axis=1)
    population_sizes = [8, 16, 32, 64, 128, 256]
    (a, b) = np.polyfit(np.log(population_sizes), np.log(y), 1)
    print(f'Fit power law: exponent={a}, coefficient={np.exp(b)}')



def clusters():

    for step in steps:
        state = torch.from_numpy(np.load(f'data/classic_big_pop_state_step_{step}.npy'))
        
        best_idx = state[..., 1].argmax(dim=-1, keepdim=True)
        top_words = state[..., 0].gather(-1, best_idx).squeeze(-1).long()[0]

        sim = (top_words[:, None, :] == top_words[None, :, :]).float().mean(-1)

        g = sns.clustermap(
            sim,
            cmap='viridis',
            linewidths=0.2,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Vocab similarity (match ratio)'},
        )
        g.ax_heatmap.set_xlabel('Agent')
        g.ax_heatmap.set_ylabel('Agent')
        g.savefig(f'plots/classic/clusters_vocab_similarity_step_{step}.png', bbox_inches='tight')
        plt.close(g.fig)
        

def agent_vocab_network(step, similarity_threshold=0.5):
    """Plot agent vocabulary similarity as network where distance = dissimilarity."""
    os.makedirs("plots/classic/population_clusters", exist_ok=True)
    
    data = torch.from_numpy(np.load(f'data/classic_big_pop_state_step_{step}.npy'))
    best_idx = data[..., 1].argmax(dim=-1, keepdim=True)
    top_words = data[..., 0].gather(-1, best_idx).squeeze(-1).long()  # (games, agents, objects)
    
    vocab = top_words[0].cpu().numpy()  # (agents, objects)
    
    # Compute pairwise similarity
    n_agents = vocab.shape[0]
    sim = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            sim[i, j] = (vocab[i] == vocab[j]).mean()
    
    # Convert similarity to distance (1 - similarity)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0)
    
    # Use MDS to embed high-dimensional distance into 2D space
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_2d = mds.fit_transform(dist)
    pos = {i: pos_2d[i] for i in range(n_agents)}
    
    # Create graph (no edges will be drawn, only nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n_agents))
    
    # Color nodes by local density (avg similarity to neighbors)
    node_colors = []
    for i in range(n_agents):
        # Average similarity to k nearest neighbors
        nearest_k = np.argsort(sim[i])[-6:-1]  # 5 nearest neighbors
        avg_sim = sim[i, nearest_k].mean()
        node_colors.append(avg_sim)
    
    # Plot
    plt.figure(figsize=(14, 12))
    
    # Draw nodes only (no edges)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=150,
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Avg similarity to neighbors')
    
    plt.title(f'Agent vocab clusters (step={step})\nPosition = vocab similarity, color = local density', 
              fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/classic/population_clusters/vocab_network_step_{step}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster visualization (step={step}):")
    print(f"  Total agents: {n_agents}")
    print(f"  Position method: MDS on vocab dissimilarity")
    print(f"  Node color: avg similarity to 5 nearest neighbors")

population_sizes = [8, 16, 32, 64, 128, 256]
object_sizes = [8, 16, 32, 64, 128, 256]
memory_sizes = [1, 2, 4, 8, 16]
vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]
context_sizes = ["(2, 2)", "(2, 3)", "(3, 4)", "(4, 5)"]
bit_flip_prob = [0.1, 0.3, 0.5, 0.7, 0.9]
obj_conf = np.linspace(0.01, 0.1, 5)
bit_flip_prob_w_obj = np.linspace(0.01, 0.1, 5)


baseline(
    'data/base_game_monte_carlo_stats.npy',
    'Parametry Bazowe',
    'baseline',
    'classic/baseline'
)

# gen('data/base_game_population_size_base.npy',
#     'Rozmiar Populacji',
#     'population_size',
#     population_sizes,
#     'classic/population_size'
#     )

# gen('data/base_game_object_size_base.npy',
#     'Rozmiar Obiektu',
#     'object_size',
#     object_sizes,
#     'classic/object_size'
#     )

# gen('data/base_game_vocab_size_base.npy',
#     'Rozmiar Słownika',
#     'vocab_size',
#     vocab_sizes,
#     'classic/vocab_size'
#     )

# gen ('data/base_game_context_size_base.npy',
#     'Rozmiar Kontekstu',
#     'context_size',
#     context_sizes,
#     'classic/context_size',
#     False
#     )


# gen('data/word_game_monte_carlo_stats.npy',
#     'Szansa Pomylenia Słowa',
#     'bit_flip_prob',
#     bit_flip_prob,
#     'word_game/baseline'
#     )

# gen('data/object_game_monte_carlo_stats.npy',
#     'Szansa Pomylenia Obiektu',
#     'obj_conf_prob',
#     obj_conf,
#     'object_game/baseline')


# gen_word_object(
#     bit_flip_prob_w_obj,
#     obj_conf,
# )

# power_law()

# steps = [0, 10, 100, 1000, 10000, 100000, 500000, 999999]


# for step in steps:
#     agent_vocab_network(step, similarity_threshold=0.5)