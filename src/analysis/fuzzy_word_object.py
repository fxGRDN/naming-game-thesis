import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import get_default_device, mean_q
from skimage import measure
from analysis.base_game import print_latex_table
from sklearn.ensemble import RandomForestRegressor



plt.figure(figsize=(6.5, 4))
plt.rcParams.update({
    "text.usetex": False,  # Set to True if LaTeX is installed
    "font.family": "serif",
})

def gen_word_object(sampling_freq: int = 100):
    device = get_default_device()

    obj_conf = torch.linspace(0, 1, 25, device=device)
    bit_flip_prob = torch.linspace(0, 1, 25, device=device)

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
            linewidths=0,
            rasterized=True,
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

    fig.supxlabel('Szansa Błędu Detekcji (p)')
    fig.supylabel('Szansa Błędu Transmisji (q)')
    plt.suptitle('Sukces interakcji - Błąd Detekcji (p) i Transmisji (q)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/success.png')
    plt.savefig(f'plots/word_object_game/success.pdf')
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
            linewidths=0,
            rasterized=True,
            ax=ax
        )

        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
        ax.invert_yaxis()
    fig.supxlabel('Szansa Błędu Detekcji (p)')
    fig.supylabel('Szansa Błędu Transmisji (q)')
    plt.suptitle('Konsensus - Błąd Detekcji (p) i Transmisji (q)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/consensus.png')
    plt.savefig(f'plots/word_object_game/consensus.pdf')
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
            linewidths=0,
            rasterized=True,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
    
    fig.supxlabel('Szansa Błędu Detekcji (p)')
    fig.supylabel('Szansa Błędu Transmisji (q)')
    plt.suptitle('Rozmiar słownika - Błąd Detekcji (p) i Transmisji (q)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/dict.png')
    plt.savefig(f'plots/word_object_game/dict.pdf')
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
            linewidths=0,
            rasterized=True,
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f'Po {(k+1)*sampling_freq} iteracjach')
    fig.supxlabel('Szansa Błędu Detekcji (p)')
    fig.supylabel('Szansa Błędu Transmisji (q)')
    plt.suptitle('Entropia - Błąd Detekcji (p) i Transmisji (q)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/word_object_game/entropy.png')
    plt.savefig(f'plots/word_object_game/entropy.pdf')
    plt.close()


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

    final_idx = 3
    
    success = STATS_2D[0, final_idx]      # higher is better [0, 1]
    consensus = STATS_2D[1, final_idx]    # higher is better [0, 1]
    dict_size = STATS_2D[2, final_idx]    # lower is better
    entropy = STATS_2D[3, final_idx]      # lower is better
    
    vocab_size = 16  # or whatever your vocab size is
    dict_size_norm = 1 - (dict_size - 1) / (vocab_size - 1)
    dict_size_norm = np.clip(dict_size_norm, 0, 1)
    
    num_objects = 16
    max_entropy = np.log2(num_objects)  # = 4.0
    entropy_norm = 1 - entropy / max_entropy
    entropy_norm = np.clip(entropy_norm, 0, 1)


    # compute CV for consensus time for each (p, q) pair
    convergence_cv = np.full((len(obj_conf), len(bit_flip_prob)), np.nan)
    for i in range(len(obj_conf)):
        try:
            data = np.load(f'data/word_object_game/monte_carlo_stats_part_{i}.npy')
            for j in range(len(bit_flip_prob)):
                consensus_trace = data[j, 1]  # shape: (time_samples, games)
                t_consensus = (consensus_trace.T > 0.90).argmax(axis=1) * sampling_freq
                mask = (consensus_trace.T > 0.90).any(axis=1)
                if mask.sum() > 10:  # need enough samples for reliable CV
                    t_valid = t_consensus[mask]
                    mean_t = t_valid.mean()
                    std_t = t_valid.std()
                    if mean_t > 0:
                        convergence_cv[i, j] = std_t / mean_t  # CV = std/mean
        except FileNotFoundError:
            pass
    
    # meaningful consensus = consensus * (1 - entropy_normalized)
    meaningful_consensus = consensus * entropy_norm
    
    valid_mask = ~np.isnan(convergence_cv)
    X_features = np.stack([
        success[valid_mask],
        meaningful_consensus[valid_mask],
        dict_size_norm[valid_mask]
    ], axis=1)
    y_target = convergence_cv[valid_mask]
    
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_features, y_target)
    
    r2_score = rf.score(X_features, y_target)
    
    print("\n" + "="*60)
    print("RANDOM FOREST ANALYSIS: Predicting CV of Consensus Time")
    print("="*60)
    print(f"R² score: {r2_score:.4f}")
    print(f"CV range: [{y_target.min():.3f}, {y_target.max():.3f}]")
    print(f"Valid samples: {valid_mask.sum()}/{valid_mask.size}")
    
    feature_names = ['Success', 'Consensus×(1-Entropy)', 'Dict Size (inverted)']
    feature_names_pl = ['Sukces interakcji', 'Skorygowany konsensus', 'Wskaźnik pamięci']
    importances = rf.feature_importances_
    
    print("\nFeature importances (Gini impurity reduction):")
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.2%}")
    
    weights = importances / importances.sum()
    print("\nDerived prosperity weights:")
    for name, w in zip(feature_names, weights):
        print(f"  {name}: {w:.2%}")
    print("="*60)
    
    print("\n% LaTeX table - Random Forest feature importances")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Wagi istotności metryk wyznaczone metodą Random Forest ($R^2 = {r2_score:.4f}$)}}")
    print("\\label{tab:rf_feature_importance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Metryka} & \\textbf{Waga istotności} \\\\")
    print("\\midrule")
    for name_pl, w in zip(feature_names_pl, weights):
        print(f"{name_pl} & {w*100:.2f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    prosperity_score = (
        weights[0] * success +
        weights[1] * meaningful_consensus +
        weights[2] * dict_size_norm
    )
    
    # normalize prosperity score to [0, 1] for better contrast
    prosperity_score_norm = (prosperity_score - prosperity_score.min()) / (prosperity_score.max() - prosperity_score.min() + 1e-8)
    
    from matplotlib.colors import LinearSegmentedColormap
    colors_rg = [(0.8, 0.2, 0.2), (1.0, 1.0, 0.4), (0.2, 0.8, 0.2)]  # red -> yellow -> green
    cmap_prosperity = LinearSegmentedColormap.from_list('prosperity', colors_rg, N=256)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        prosperity_score_norm.T,
        origin='lower',
        extent=(obj_conf.min(), obj_conf.max(), bit_flip_prob.min(), bit_flip_prob.max()),
        aspect='auto',
        cmap=cmap_prosperity,
        vmin=0, vmax=1
    )
    
    cbar = plt.colorbar(im, ax=ax, label='Miara jakości języka (znormalizowana)')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    
    ax.set_xlabel('Szansa błędu detekcji (p)', fontsize=12)
    ax.set_ylabel('Szansa błędu transmisji (q)', fontsize=12)
    ax.set_title(f'Regiony sprzyjające rozwojowi języka\n(po 100 000 iteracji, RF $R^2$={r2_score:.3f})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('plots/word_object_game/language_prosperity.png', dpi=150)
    plt.savefig('plots/word_object_game/language_prosperity.pdf')
    plt.close()
    
    print(f"\nProsperity score range: [{prosperity_score.min():.3f}, {prosperity_score.max():.3f}]")
    print("Saved language prosperity plot to plots/word_object_game/language_prosperity.png")

    print_latex_table_pairs(obj_conf, bit_flip_prob, sampling_freq)


def print_latex_table_pairs(obj_conf, bit_flip_prob, sampling_freq=100):
    """print LaTeX table with statistics for 4 parameter pairs."""
    
    os.makedirs("data/word_object_game/single", exist_ok=True)

    if isinstance(obj_conf, torch.Tensor):
        obj_conf = obj_conf.cpu().numpy()
    if isinstance(bit_flip_prob, torch.Tensor):
        bit_flip_prob = bit_flip_prob.cpu().numpy()
    
    target_pairs = [
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    ]
    
    pair_indices = []
    for p_target, q_target in target_pairs:
        p_idx = np.abs(obj_conf - p_target).argmin()
        q_idx = np.abs(bit_flip_prob - q_target).argmin()
        pair_indices.append((p_idx, q_idx))
    
    def fmt_time(mean, std):
        if np.isnan(mean) or np.isnan(std):
            return "--"
        if mean == 0:
            return "$0$"
        exponent = int(np.floor(np.log10(max(mean, 1))))
        if exponent >= 3:
            exp_display = (exponent // 3) * 3
            scale = 10 ** exp_display
            return f"${mean/scale:.1f} \\pm {std/scale:.1f}$"
        else:
            return f"${mean:.0f} \\pm {std:.0f}$"
    
    stats = {
        'max_dict': [],
        'time_max_dict': [],
        't_success': [],
        't_consensus': [],
        'stable_entropy': [],
        't_stable_entropy': [],
        't_stable_dict': []
    }
    
    for q_idx, p_idx in pair_indices:
        try:
            full_data = np.load(f"data/word_object_game/monte_carlo_stats_part_{q_idx}.npy")
            data = full_data[p_idx]
            
            t_success_all = (data[0].T > 0.90).argmax(axis=1)
            mask_success = t_success_all != 0
            t_success = t_success_all[mask_success] * sampling_freq
            
            t_consensus_all = (data[1].T > 0.90).argmax(axis=1)
            mask_consensus = t_consensus_all != 0
            t_consensus = t_consensus_all[mask_consensus] * sampling_freq
            
            max_dict = data[2].max(axis=0)
            time_to_max_dict = data[2].T.argmax(axis=1)
            mask = time_to_max_dict != 0
            time_to_max_dict_filtered = time_to_max_dict[mask] * sampling_freq
            
            tolerance = 0.05
            t_entropy = []
            for sim in range(data[3].shape[1]):
                start_idx = time_to_max_dict[sim]
                entropy_sim = data[3][start_idx:, sim]
                if len(entropy_sim) > 0:
                    final_entropy = entropy_sim[-1]
                    threshold = max(tolerance * final_entropy, 0.02)
                    stable_idx = np.where(np.abs(entropy_sim - final_entropy) < threshold)[0]
                    if len(stable_idx) > 0:
                        t_entropy.append((start_idx + stable_idx[0]) * sampling_freq)
            t_entropy = np.array(t_entropy) if t_entropy else np.array([np.nan])
            
            t_stable_dict = []
            for sim in range(data[2].shape[1]):
                start_idx = time_to_max_dict[sim]
                dict_sim = data[2][start_idx:, sim]
                stable_idx = np.where(dict_sim <= 1.0)[0]
                if len(stable_idx) > 0:
                    t_stable_dict.append((start_idx + stable_idx[0]) * sampling_freq)
            t_stable_dict = np.array(t_stable_dict) if t_stable_dict else np.array([np.nan])
            
            stable_entropy = data[3][-1, :]
            
            stats['max_dict'].append((max_dict.mean(), max_dict.std()))
            stats['time_max_dict'].append((time_to_max_dict_filtered.mean() if len(time_to_max_dict_filtered) > 0 else np.nan, 
                                           time_to_max_dict_filtered.std() if len(time_to_max_dict_filtered) > 0 else np.nan))
            stats['t_success'].append((t_success.mean() if len(t_success) > 0 else np.nan, 
                                        t_success.std() if len(t_success) > 0 else np.nan))
            stats['t_consensus'].append((t_consensus.mean() if len(t_consensus) > 0 else np.nan, 
                                          t_consensus.std() if len(t_consensus) > 0 else np.nan))
            stats['stable_entropy'].append((stable_entropy.mean(), stable_entropy.std()))
            stats['t_stable_entropy'].append((np.nanmean(t_entropy), np.nanstd(t_entropy)))
            stats['t_stable_dict'].append((np.nanmean(t_stable_dict), np.nanstd(t_stable_dict)))
            
            del full_data
        except FileNotFoundError:
            for key in stats:
                stats[key].append((np.nan, np.nan))
            
    print("\n" + "%"*60)
    print("% LaTeX table - Word+Object confusion probability analysis")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Statystyki symulacji vs kombinacje szans pomylenia}")
    print("\\label{tab:word_object_confusion_stats}")
    
    n_cols = len(pair_indices)
    col_spec = "l" + "c" * n_cols
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    header = "\\textbf{Metryka} & " + " & ".join([f"\\shortstack{{$p={obj_conf[p_idx]:.2f}$\\\\$q={bit_flip_prob[q_idx]:.2f}$}}" for p_idx, q_idx in pair_indices]) + " \\\\"
    print(header)
    print("\\midrule")
    
    metrics = [
        ('Maks. rozmiar słownika', 'max_dict', False),
        ('Czas do maks. słownika ($\\times 10^3$)', 'time_max_dict', True),
        ('Czas do 90\\% sukcesu ($\\times 10^3$)', 't_success', True),
        ('Czas do 90\\% konsensusu ($\\times 10^3$)', 't_consensus', True),
        ('Stabilna entropia', 'stable_entropy', False),
        ('Czas do stab. entropii ($\\times 10^3$)', 't_stable_entropy', True),
        ('Czas do stab. słownika ($\\times 10^3$)', 't_stable_dict', True),
    ]
    
    for label, key, is_time in metrics:
        if is_time:
            row = f"{label} & " + " & ".join([fmt_time(m, s) for m, s in stats[key]]) + " \\\\"
        else:
            row = f"{label} & " + " & ".join([f"${m:.2f} \\pm {s:.2f}$" if not np.isnan(m) else "--" for m, s in stats[key]]) + " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("%"*60 + "\n")
    
    print("\n" + "="*60)
    print("Statistical Significance Tests (Mann-Whitney U)")
    print("="*60)
    


def gen_word_object_single(confusion_prob, flip_prob, sampling_freq):

    # (stats, samples, games)
    data_1_0 = np.load("data/word_object_game/single/monte_carlo_stats_part_" +
                       f"{confusion_prob}_{flip_prob}.npy", allow_pickle=True)
    print(data_1_0.shape)
    x = np.arange(data_1_0.shape[-2]) * sampling_freq
    print(x.max())
    plt.figure(figsize=(12, 6))

    os.makedirs(f'plots/word_object_game/single', exist_ok=True)


    mean, lo, hi = mean_q(data_1_0[0])

    plt.subplot(1, 2, 1)
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title(f'Średni sukces w czasie - (p,q) = $({confusion_prob},{flip_prob})$')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.legend()
    plt.subplot(1, 2, 2)

    time_dist = (data_1_0[0].T > 0.90).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]*sampling_freq

    print(time_dist.mean(), time_dist.std())

    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 90% sukcesu - {confusion_prob}_{flip_prob}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')

    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_success_rate.png')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_success_rate.pdf')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    mean, lo, hi = mean_q(data_1_0[1])

    for i in np.random.randint(0, data_1_0[1].shape[1], size=5):
        sns.lineplot(x=x, y=data_1_0[1].T[i], alpha=0.3, color='gray', label='_nolegend_')

    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title(f'Średni konsensus w czasie - (p,q) = $({confusion_prob},{flip_prob})$')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Konsensus')
    plt.legend()
    plt.subplot(1, 2, 2)
    time_dist = (data_1_0[1].T > 0.90).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]*sampling_freq


    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 90% konsensusu - (p,q) = $({confusion_prob},{flip_prob})$')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_consensus.png')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_consensus.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[2])


    time_dist = data_1_0[2].T.argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]*sampling_freq


    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title('Średni rozmiar słownika w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.legend()
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_dict_size.png')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_dict_size.pdf')
    plt.close()


    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[3])
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.hlines(0, xmin=0, xmax=len(x)*sampling_freq-1, colors='r', linestyles='dashed', label='Brak entropii')
    plt.title(f'Entropia referencyjna - (p,q) = $({confusion_prob},{flip_prob})$')
    plt.legend()
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_entropy.png')
    plt.savefig(f'plots/word_object_game/single/{confusion_prob}_{flip_prob}_analysis_entropy.pdf')
    plt.close()

    print_latex_table(data_1_0, f'(p,q) = $({confusion_prob},{flip_prob})$', f'{confusion_prob}_{flip_prob}', sampling_freq)

    del data_1_0







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    parser.add_argument("-p", "--confusion-prob", type=float, help="Object confusion probability for single run", default=None)
    parser.add_argument("-q", "--flip-prob", type=float, help="Word bit flip probability for single run", default=None)
    parser.add_argument("--single", action="store_true", help="Run single simulation instead of monte-carlo sweep")
    args = parser.parse_args()


    if args.single:
        if args.confusion_prob is None or args.flip_prob is None:
            raise ValueError("For single run, both --confusion-prob and --flip-prob must be specified.")
        gen_word_object_single(args.confusion_prob, args.flip_prob, args.sampling_freq)
    else:   
        gen_word_object(sampling_freq=args.sampling_freq)
