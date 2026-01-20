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

def entropy_law():
    data = np.load('data/base_game_vocab_size_base.npy')
    y = data[:, 3].mean(axis=-1)[:, -1]
    vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]
    (a, b, c) = np.polyfit(np.log(vocab_sizes), np.log(y), 2)

    curve = lambda x, a, b: a + b/x

    (a,b), pcov = scipy.optimize.curve_fit(curve, vocab_sizes, y)



    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, y, label='Dane')
    plt.plot(vocab_sizes, curve(np.array(vocab_sizes), a, b), '--', label=f'Dopasowanie: y={a:.2f} + {b:.2f}/x')
    plt.xlabel('Rozmiar Słownika')
    plt.ylabel('Średnia Entropia Referencyjna')
    plt.title('Entropia referencyjna w zależności od rozmiaru słownika')
    plt.legend()
    plt.savefig('plots/classic/entropy_law_vocab_size.png')
    plt.close()

    # print(f'Fit entropy law: exponent={a}, coefficient={np.exp(b)}')



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


# baseline(
#     'data/base_game_monte_carlo_stats.npy',
#     'Parametry Bazowe',
#     'baseline',
#     'classic/baseline'
# )

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


gen('data/word_game_monte_carlo_stats.npy',
    'Szansa Pomylenia Słowa',
    'bit_flip_prob',
    bit_flip_prob,
    'word_game/baseline'
    )

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
# entropy_law()


# steps = [0, 10, 100, 1000, 10000, 100000, 500000, 999999]


# for step in steps:
#     agent_vocab_network(step, similarity_threshold=0.5)