import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import success_rate_ma, mean_q
import scipy.stats

sns.set_theme(style="whitegrid")


def baseline(path, title_prefix, label_suffix, folder, sampling_freq=100):

    # (stats, samples, games)
    data_1_0 = np.load(path)
    print(data_1_0.shape)
    x = np.arange(data_1_0.shape[-2]) * sampling_freq
    print(x.max())
    plt.figure(figsize=(12, 6))

    os.makedirs(f'plots/{folder}', exist_ok=True)


    mean, lo, hi = mean_q(data_1_0[0])

    plt.subplot(1, 2, 1)
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.title(f'Średni sukces w czasie - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.legend()
    plt.subplot(1, 2, 2)

    time_dist = (data_1_0[0].T > 0.90).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]*sampling_freq

    print(time_dist.mean(), time_dist.std())

    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 90% sukcesu - {title_prefix}')
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
    plt.title(f'Średni konsensus w czasie - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Konsensus')
    plt.legend()
    plt.subplot(1, 2, 2)
    time_dist = (data_1_0[1].T > 0.90).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]*sampling_freq


    sns.histplot(time_dist, kde=False)
    plt.title(f'Czas do osiągnięcia 90% konsensusu - {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_consensus.png')
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
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_dict_size.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[3])
    sns.lineplot(x=x, y=mean)
    plt.fill_between(x, lo, hi, alpha=0.3, label=" 99% przedział tolerancji")
    plt.hlines(0, xmin=0, xmax=len(x)*sampling_freq-1, colors='r', linestyles='dashed', label='Brak entropii')
    plt.title(f'Entropia referencyjna - {title_prefix}')
    plt.legend()
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_entropy.png')
    plt.close()

    print_latex_table(data_1_0, title_prefix, label_suffix, sampling_freq)

    del data_1_0


def print_latex_table(data, title_prefix, label_suffix, sampling_freq=100):
    """Print LaTeX table with simulation statistics."""
    
    mean_success_ma = success_rate_ma(data[0])
    t_success_all = (mean_success_ma.T > 0.90).argmax(axis=1)
    mask_success = t_success_all != 0
    t_success = t_success_all[mask_success] * sampling_freq
    
    t_consensus_all = (data[1].T > 0.90).argmax(axis=1)
    mask_consensus = t_consensus_all != 0
    t_consensus = t_consensus_all[mask_consensus] * sampling_freq
    
    if len(t_consensus) > 10:
        # Fit parameters
        mean_cons, std_cons = t_consensus.mean(), t_consensus.std()
        loc_laplace, scale_laplace = scipy.stats.laplace.fit(t_consensus)
        
        ks_norm, p_norm = scipy.stats.kstest(t_consensus, 'norm', args=(mean_cons, std_cons))
        ks_laplace, p_laplace = scipy.stats.kstest(t_consensus, 'laplace', args=(loc_laplace, scale_laplace))
        
        print(f"\nConsensus time distribution test:")
        print(f"  Gaussian:  KS={ks_norm:.4f}, p={p_norm:.4e}")
        print(f"  Laplace:   KS={ks_laplace:.4f}, p={p_laplace:.4e}")
        print(f"  Better fit: {'Laplace' if p_laplace > p_norm else 'Gaussian'}")
    
    common_mask = mask_success & mask_consensus
    if common_mask.sum() > 1:
        corr = np.corrcoef(t_success_all[common_mask], t_consensus_all[common_mask])[0, 1]
        print(f"Correlation (success vs consensus time): {corr:.4f}")
    
    max_dict = data[2].max(axis=0)
    time_to_max_dict = data[2].T.argmax(axis=1)
    mask = time_to_max_dict != 0
    time_to_max_dict_filtered = time_to_max_dict[mask] * sampling_freq
    mean_max_time = time_to_max_dict_filtered.mean()
    std_max_time = time_to_max_dict_filtered.std()

    tolerance = 0.05 
    
    t_entropy = []
    for sim in range(data[3].shape[1]):
        start_idx = time_to_max_dict[sim]  
        entropy_sim = data[3][start_idx:, sim]
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
    

    
    def fmt_time(mean, std):
        if np.isnan(mean) or np.isnan(std):
            return "--"
        if mean == 0:
            return "$0$"
        exponent = int(np.floor(np.log10(mean)))
        if exponent >= 3:
            exp_display = (exponent // 3) * 3
            scale = 10 ** exp_display
            return f"${mean/scale:.2f} \\pm {std/scale:.2f}$ ($\\times 10^{{{exp_display}}}$)"
        else:
            return f"${mean:.1f} \\pm {std:.1f}$"
    
    print("\n" + "%"*60)
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Statystyki symulacji -- {title_prefix}}}")
    print(f"\\label{{tab:{label_suffix}_stats}}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Metryka} & \\textbf{Wartość (średnia $\\pm$ odch. std.)} \\\\")
    print("\\midrule")
    
    print(f"Maksymalny rozmiar słownika & ${max_dict.mean():.2f} \\pm {max_dict.std():.2f}$ \\\\")
    print(f"Czas do maks. rozmiaru słownika & {fmt_time(mean_max_time, std_max_time)} \\\\")

    if len(t_success) > 0:
        print(f"Czas do 90\\% sukcesu interakcji & {fmt_time(t_success.mean(), t_success.std())} \\\\")
    else:
        print("Czas do 90\\% sukcesu interakcji & -- \\\\")
    
    if len(t_consensus) > 0:
        print(f"Czas do 90\\% konsensusu & {fmt_time(t_consensus.mean(), t_consensus.std())} \\\\")
    else:
        print("Czas do 90\\% konsensusu & -- \\\\")
    
    stable_entropy = data[3][-1, :]  # Entropy at final time step
    stable_homonymy = np.power(2, stable_entropy)
    print(f"Stabilna entropia & ${stable_homonymy.mean():.2f} \\pm {stable_homonymy.std():.2f}$ \\\\")
    
    if len(t_entropy) > 0 and not np.isnan(t_entropy).all():
        print(f"Czas do stabilnej entropii & {fmt_time(np.nanmean(t_entropy), np.nanstd(t_entropy))} \\\\")
    else:
        print("Czas do stabilnej entropii & -- \\\\")
    
    print(f"Czas do stabilnego słownika & {fmt_time(np.nanmean(t_stable_dict), np.nanstd(t_stable_dict))} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("%"*60 + "\n")

def baseline_multiple(path, title_prefix, label_suffix, param_values, folder, log_plot=False, sampling_freq=100):
    os.makedirs(f'plots/{folder}', exist_ok=True)
    data = np.load(path)

    param_labels = [str(p) for p in param_values]

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    print("Analyzing baseline multiple:", label_suffix)

    # mean success rate
    ma_stat_0 = success_rate_ma(data[:, 0])
    for stat, param in zip(ma_stat_0, param_labels):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2])*sampling_freq, y=stat.mean(axis=-1), label = param)
        plt.subplot(1, 2, 2)
        time_dist = (stat.T > 0.90).argmax(axis=1)
        mask = time_dist != 0
        time_dist = time_dist[mask] * sampling_freq
        time_dist_df = pd.DataFrame({label_suffix: [param]*len(time_dist), 'time_to_90_success': time_dist})
        if len(time_dist) > 0:
            sns.boxplot(data=time_dist_df, y='time_to_90_success', x=label_suffix)
    
    plt.subplot(1, 2, 1)
    plt.title(f'Średni sukces w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')

    plt.subplot(1, 2, 2)
    plt.title(f'Czas do osiągnięcia 90% sukcesu vs {title_prefix}')
    plt.ylabel('Kroki Symulacji')
    plt.xlabel(title_prefix)

    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_success_rate.png')
    plt.close()

    ####################################################
    ####################################################
    ####################################################

    plt.figure(figsize=(13, 6))
    plt.tight_layout()

    for stat, param in zip(data[:, 1], param_labels):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2]) * sampling_freq, y=stat.mean(axis=-1), label = param)
        plt.subplot(1, 2, 2)
        time_dist = (stat.T > 0.90).argmax(axis=1)
        mask = time_dist != 0
        time_dist = time_dist[mask] * sampling_freq
        time_dist_df = pd.DataFrame({label_suffix: [param]*len(time_dist), 'time_to_90_success': time_dist})

        if len(time_dist) > 0:
            sns.boxplot(data=time_dist_df, y='time_to_90_success', x=label_suffix)
    
    plt.subplot(1, 2, 1)
    plt.title(f'Średni konsensus vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni konsensus')

    plt.subplot(1, 2, 2)
    plt.title(f'Czas do osiągnięcia 90% konsensusu vs {title_prefix}')
    plt.ylabel('Kroki Symulacji')
    plt.xlabel(title_prefix)

    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_consensus.png')
    plt.close()

    ####################################################
    ####################################################
    ####################################################


    plt.figure(figsize=(13, 6))
    for stat, param in zip(data[:, 2], param_labels):
        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(stat.shape[-2]) * sampling_freq, y=stat.mean(axis=-1), label = param)

        plt.subplot(1, 2, 2)
        # czas na osiagniecie peaku
        peaks = stat.T.argmax(axis=1) * sampling_freq
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
        y = data[:, 2].transpose(0, 2, 1).argmax(axis=2).mean(axis=1) * sampling_freq
        plt.loglog(param_values, y, )
        plt.loglog(param_values, np.array(param_values)**(1.5), '--', label='Oczekiwana złożoność $O(n^{1.5})$')
        plt.title(f'Średni czas do osiągnięcia maksymalnego rozmiaru słownika vs {title_prefix}')
        plt.xlabel(title_prefix)
        plt.ylabel('Kroki Symulacji')
        plt.legend()

        plt.savefig(f'plots/{folder}/{label_suffix}_analysis_dict_size_mean_time.png')
        plt.close()

    for stat, param in zip(data[: ,3], param_labels):
        sns.lineplot(x=np.arange(stat.shape[-2]) * sampling_freq, y=stat.mean(axis=-1), label = param)

    plt.legend()

    plt.title(f'Średnia entropia referencyjna w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia Entropia Referencyjna')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_entropy.png')
    plt.close()

    del data


# )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", action="store_true",  help="Run baseline analysis")
    parser.add_argument("-p", action="store_true",  help="Run population size analysis")
    parser.add_argument("-o", action="store_true",  help="Run object size analysis")
    parser.add_argument("-v", action="store_true",  help="Run vocab size analysis")
    parser.add_argument("-c", action="store_true",  help="Run context window size analysis")
    parser.add_argument("-m", action="store_true",  help="Run memory size analysis")
    parser.add_argument("-s", "--sampling-freq", type=int, help="Sampling frequency for analysis, default 100", nargs='?', default=100)
    args = parser.parse_args()

    if args.b:
        baseline(
            "data/baseline/baseline.npy", 
            "Parametery bazowe", 
            "baseline", 
            "classic/baseline",
            args.sampling_freq
        )
    if args.p:
        baseline_multiple(
            "data/baseline/population_size.npy",
            "Rozmiar populacji",
            "population_size",
            [8, 16, 32, 64, 128, 256],
            "classic/population_size",
            log_plot=True,
            sampling_freq=args.sampling_freq
        )

    if args.o:
        baseline_multiple(
            "data/baseline/object_size.npy",
            "Liczba obiektów",
            "object_size",
            [8, 16, 32, 64, 128, 256],
            "classic/object_size",
            sampling_freq=args.sampling_freq
        )

    if args.v:
        baseline_multiple(
            "data/baseline/vocab_size.npy",
            "Rozmiar słownika",
            "vocab_size",
            [2**4, 2**6, 2**8, 2**10, 2**12],
            "classic/vocab_size",
            sampling_freq=args.sampling_freq
        )

    if args.c:
        baseline_multiple(
            "data/baseline/context_size.npy",
            "Rozmiar kontekstu",
            "context_size",
            [(1, 2), (2, 4), (4, 6), (6, 8)],
            "classic/context_size",
            sampling_freq=args.sampling_freq
            )
    if args.m:
        baseline_multiple(
            "data/baseline/memory_size.npy",
            "Rozmiar pamięci",
            "memory_size",
            [3, 5, 7, 8, 12, 16],
            "classic/memory_size",
            sampling_freq=args.sampling_freq
            )