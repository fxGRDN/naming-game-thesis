import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats
import pandas as pd
from utils import success_rate_ma, mean_q


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
          "\n Last value", mean[-1],
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
          "\n Last value", mean[-1],
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

    print(mean.max())

    time_dist = data_1_0[2].T.argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]
    print(time_dist.mean())


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



def baseline_multiple(path, title_prefix, label_suffix, param_values, folder, log_plot=True):
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

    plt.legend()

    plt.title(f'Średnia entropia referencyjna w czasie vs {title_prefix}')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia Entropia Referencyjna')
    plt.savefig(f'plots/{folder}/{label_suffix}_analysis_entropy.png')
    plt.close()

