import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils import success_rate_ma
from simulations.parameters import metrics_limits


def consensus_threshold(sampling_freq=100):
    bit_flip_prob = np.linspace(0, 1, 100)
    y = np.zeros((len(bit_flip_prob), 4, 100000 // sampling_freq))
    try:
        for i, p in enumerate(bit_flip_prob):
            data = np.load(f"data/object_phase/part_{i}.npy")
            print(data.shape)
            y[i] = data.mean(axis=-1)
            if i == 0:
                y[i] = success_rate_ma(data[0]).mean(axis=-1)
            del data

    except FileNotFoundError:
        pass



    os.makedirs("plots/object_game", exist_ok=True)
    sample_idx = np.arange(y.shape[2]) * sampling_freq

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
        ax.set_zlim(metrics_limits[stat])

        ax.view_init(elev=35.264, azim=45)  # isometric-like view

        im = axhm.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=(bit_flip_prob.min(), bit_flip_prob.max(), sample_idx.min(), sample_idx.max()),
            cmap="viridis",
            vmin=metrics_limits[stat][0],
            vmax=metrics_limits[stat][1],
        )
        axhm.set_xlabel("Szansa Pomylenia Obiektu")
        axhm.set_ylabel("Indeks Próby")
        fig.colorbar(im, ax=axhm, shrink=0.8)

        fig.suptitle(stat_names[stat])
        plt.tight_layout()
        plt.savefig(f"plots/object_game/surface_stat_{stat}.png", dpi=200)
        plt.close(fig)

    print_latex_table_multiple(bit_flip_prob, sampling_freq)


def print_latex_table_multiple(param_values, sampling_freq=100):
    """Print LaTeX table with statistics for multiple parameter values."""
    
    target_values = [0, 0.3, 0.5, 0.7]
    param_indices = [np.abs(param_values - t).argmin() for t in target_values]
    
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
    
    for i in param_indices:
        try:
            data = np.load(f"data/object_phase/part_{i}.npy")
            
            mean_success_ma = success_rate_ma(data[0], window_size=10)
            t_success_all = (mean_success_ma.T > 0.90).argmax(axis=1)
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
            
            del data
        except FileNotFoundError:
            for key in stats:
                stats[key].append((np.nan, np.nan))
    
    # Print LaTeX table
    print("\n" + "%"*60)
    print("% LaTeX table - Object confusion probability analysis")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Statystyki symulacji vs szansa pomylenia obiektu}")
    print("\\label{tab:object_confusion_stats}")
    
    # Header
    n_cols = len(param_indices)
    col_spec = "l" + "c" * n_cols
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    header = "\\textbf{Metryka} & " + " & ".join([f"$p={param_values[i]:.2f}$" for i in param_indices]) + " \\\\"
    print(header)
    print("\\midrule")
    
    # Rows
    metrics = [
        ('Maks. rozmiar słownika', 'max_dict', False),
        ('Czas do maks. słownika', 'time_max_dict', True),
        ('Czas do 90\\% sukcesu', 't_success', True),
        ('Czas do 90\\% konsensusu', 't_consensus', True),
        ('Stabilna entropia', 'stable_entropy', False),
        ('Czas do stab. entropii', 't_stable_entropy', True),
        ('Czas do stab. słownika', 't_stable_dict', True),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis", default=100)
    args = parser.parse_args()
    
    os.makedirs("data/object_phase", exist_ok=True)
    consensus_threshold(sampling_freq=args.sampling_freq)