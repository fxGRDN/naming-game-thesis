import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# data_1_2 = np.load('data/base_game_object_size_consensus.npy')
# data_1_3 = np.load('data/base_game_memory_size_consensus.npy')
# data_1_4 = np.load('data/base_game_vocab_size_consensus.npy')

# mean_1_2 = np.mean(data_1_2, axis=-1)
# mean_1_3 = np.mean(data_1_3, axis=-1)
# mean_1_4 = np.mean(data_1_4, axis=-1)


def success_rate_ma(x, window_size=100):
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




def baseline():
    data_1_0 = np.load('data/base_game_monte_carlo_stats.npy')
    x = np.arange(data_1_0.shape[-2])
    plt.figure(figsize=(12, 6))

    mean_success_ma = success_rate_ma(data_1_0[0])

    mean, lo, hi = mean_q(mean_success_ma)

    plt.subplot(1, 2, 1)
    sns.lineplot(x=x, y=mean, label = 'Średni Sukces')
    plt.fill_between(x, lo, hi, alpha=0.3)
    plt.title('Średni sukces w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')

    plt.subplot(1, 2, 2)

    time_dist = (mean_success_ma.T > 0.99).argmax(axis=1)
    sns.histplot(time_dist, kde=False)
    plt.title('Czas do osiągnięcia 99% sukcesu - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')

    plt.savefig('plots/baseline/baseline_success_rate.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    mean, lo, hi = mean_q(data_1_0[1])

    for i in np.random.randint(0, data_1_0[1].shape[1], size=5):
        sns.lineplot(x=x, y=data_1_0[1].T[i], alpha=0.3, color='gray', label='_nolegend_')

    sns.lineplot(x=x, y=mean, label = 'Średnia Spójność')
    plt.fill_between(x, lo, hi, alpha=0.3)
    plt.title('Średnia spójność w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia Spójność')

    plt.subplot(1, 2, 2)
    time_dist = (data_1_0[1].T > 0.99).argmax(axis=1)
    mask = time_dist != 0
    time_dist = time_dist[mask]

    sns.histplot(time_dist, kde=False)
    plt.title('Czas do osiągnięcia 99% spójności - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Liczba Symulacji')

    plt.savefig('plots/baseline/baseline_consensus.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[2])
    sns.lineplot(x=x, y=mean, label = 'Średni Rozmiar Słownika')
    plt.fill_between(x, lo, hi, alpha=0.3)
    plt.title('Średni rozmiar słownika w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/baseline/baseline_dict_size.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    mean, lo, hi = mean_q(data_1_0[3])
    sns.lineplot(x=x, y=mean, label = 'Średni Wskaznik Homonimów')
    plt.fill_between(x, lo, hi, alpha=0.3)
    plt.hlines(1, xmin=0, xmax=len(x)-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie - Parametry Bazowe')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/baseline/baseline_homonymy_ratio.png')
    plt.close()



def pop_size():
    data_1_1 = np.load('data/base_game_population_size_consensus.npy')
    mean_1_1 = np.mean(data_1_1, axis=-1)
    pop_sizes = [8, 16, 32, 64, 128, 256, 512]

    plt.figure(figsize=(10, 6))

    for pop_data, pop_size in zip(mean_1_1, pop_sizes):
        sns.lineplot(x=np.arange(mean_1_1.shape[-1]), y=pop_data[0], label = pop_size)
    plt.title('Średni sukces w czasie vs Wielkość Populacji')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.savefig('plots/classic/pop_size_analysis_success_rate.png')
    plt.cla()


    for pop_data, pop_size in zip(mean_1_1, pop_sizes):
        sns.lineplot(x=np.arange(mean_1_1.shape[-1]), y=pop_data[1], label = pop_size)
    plt.title('Średnia spójność vs Wielkość Populacji')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')
    plt.savefig('plots/classic/pop_size_analysis_consensus.png')
    plt.cla()


    for pop_data, pop_size in zip(mean_1_1, pop_sizes):
        sns.lineplot(x=np.arange(mean_1_1.shape[-1]), y=pop_data[2], label = pop_size)
    plt.title('Średni rozmiar słownika w czasie vs Wielkość Populacji')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/classic/pop_size_analysis_dict_size.png')
    plt.cla()


    for pop_data, pop_size in zip(mean_1_1, pop_sizes):
        sns.lineplot(x=np.arange(mean_1_1.shape[-1]), y=pop_data[3], label = pop_size)

    plt.hlines(1, xmin=0, xmax=mean_1_1.shape[-1]-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie vs Wielkość Populacji')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/classic/pop_size_analysis_homonymy_ratio.png')
    plt.cla()


def obj_size():
    data_1_2 = np.load('data/base_game_object_size_consensus.npy')
    mean_1_2 = np.mean(data_1_2, axis=-1)
    obj_sizes = [8, 16, 32, 64, 128, 256, 512]

    plt.figure(figsize=(10, 6))

    for obj_data, obj_size in zip(mean_1_2, obj_sizes):
        sns.lineplot(x=np.arange(mean_1_2.shape[-1]), y=obj_data[0], label = obj_size)
    plt.title('Średni sukces w czasie vs Rozmiar Obiektu')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.savefig('plots/classic/obj_size_analysis_success_rate.png')
    plt.cla()


    for obj_data, obj_size in zip(mean_1_2, obj_sizes):
        sns.lineplot(x=np.arange(mean_1_2.shape[-1]), y=obj_data[1], label = obj_size)
    plt.title('Średnia spójność vs Rozmiar Obiektu')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')
    plt.savefig('plots/classic/obj_size_analysis_consensus.png')
    plt.cla()


    for obj_data, obj_size in zip(mean_1_2, obj_sizes):
        sns.lineplot(x=np.arange(mean_1_2.shape[-1]), y=obj_data[2], label = obj_size)
    plt.title('Średni rozmiar słownika w czasie vs Rozmiar Obiektu')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/classic/obj_size_analysis_dict_size.png')
    plt.cla()


    for obj_data, obj_size in zip(mean_1_2, obj_sizes):
        sns.lineplot(x=np.arange(mean_1_2.shape[-1]), y=obj_data[3], label = obj_size)

    plt.hlines(1, xmin=0, xmax=mean_1_2.shape[-1]-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie vs Rozmiar Obiektu')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/classic/obj_size_analysis_homonymy_ratio.png')
    plt.cla()


def mem_size():
    data_1_3 = np.load('data/base_game_memory_size_consensus.npy')
    mean_1_3 = np.mean(data_1_3, axis=-1)
    mem_sizes = [1, 5, 10, 20, 50, 100]

    plt.figure(figsize=(10, 6))

    for mem_data, mem_size in zip(mean_1_3, mem_sizes):
        sns.lineplot(x=np.arange(mean_1_3.shape[-1]), y=mem_data[0], label = mem_size)
    plt.title('Średni sukces w czasie vs Rozmiar Pamięci')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.savefig('plots/classic/mem_size_analysis_success_rate.png')
    plt.cla()


    for mem_data, mem_size in zip(mean_1_3, mem_sizes):
        sns.lineplot(x=np.arange(mean_1_3.shape[-1]), y=mem_data[1], label = mem_size)
    plt.title('Średnia spójność vs Rozmiar Pamięci')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')
    plt.savefig('plots/classic/mem_size_analysis_consensus.png')
    plt.cla()


    for mem_data, mem_size in zip(mean_1_3, mem_sizes):
        sns.lineplot(x=np.arange(mean_1_3.shape[-1]), y=mem_data[2], label = mem_size)
    plt.title('Średni rozmiar słownika w czasie vs Rozmiar Pamięci')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/classic/mem_size_analysis_dict_size.png')
    plt.cla()


    for mem_data, mem_size in zip(mean_1_3, mem_sizes):
        sns.lineplot(x=np.arange(mean_1_3.shape[-1]), y=mem_data[3], label = mem_size)

    plt.hlines(1, xmin=0, xmax=mean_1_3.shape[-1]-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie vs Rozmiar Pamięci')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/classic/mem_size_analysis_homonymy_ratio.png')
    plt.cla()


def vocab_size():
    data_1_4 = np.load('data/base_game_vocab_size_consensus.npy')
    mean_1_4 = np.mean(data_1_4, axis=-1)
    vocab_sizes = [2**4, 2**6, 2**8, 2**10, 2**12]

    plt.figure(figsize=(10, 6))

    for vocab_data, vocab_size in zip(mean_1_4, vocab_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=vocab_data[0], label = vocab_size)
    plt.title('Średni sukces w czasie vs Rozmiar Słownika')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.savefig('plots/classic/vocab_size_analysis_success_rate.png')
    plt.cla()


    for vocab_data, vocab_size in zip(mean_1_4, vocab_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=vocab_data[1], label = vocab_size)
    plt.title('Średnia spójność vs Rozmiar Słownika')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')
    plt.savefig('plots/classic/vocab_size_analysis_consensus.png')
    plt.cla()


    for vocab_data, vocab_size in zip(mean_1_4, vocab_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=vocab_data[2], label = vocab_size)
    plt.title('Średni rozmiar słownika w czasie vs Rozmiar Słownika')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/classic/vocab_size_analysis_dict_size.png')
    plt.cla()


    for vocab_data, vocab_size in zip(mean_1_4, vocab_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=vocab_data[3], label = vocab_size)

    plt.hlines(1, xmin=0, xmax=mean_1_4.shape[-1]-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie vs Rozmiar Słownika')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/classic/vocab_size_analysis_homonymy_ratio.png')
    plt.cla()

def context_size():
    data_1_4 = np.load('data/base_game_context_size_consensus.npy')
    mean_1_4 = np.mean(data_1_4, axis=-1)
    context_sizes = [(2, 2), (2, 3), (3, 4), (4, 5)]

    plt.figure(figsize=(10, 6))

    for ctx_data, ctx_size in zip(mean_1_4, context_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=ctx_data[0], label = f"{ctx_size[0]}x{ctx_size[1]}")
    plt.title('Średni sukces w czasie vs Rozmiar Okna Kontekstowego')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Sukces')
    plt.savefig('plots/classic/context_size_analysis_success_rate.png')
    plt.cla()


    for ctx_data, ctx_size in zip(mean_1_4, context_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=ctx_data[1], label = f"{ctx_size[0]}x{ctx_size[1]}")
    plt.title('Średnia spójność vs Rozmiar Okna Kontekstowego')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średnia spójność')
    plt.savefig('plots/classic/context_size_analysis_consensus.png')
    plt.cla()


    for ctx_data, ctx_size in zip(mean_1_4, context_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=ctx_data[2], label = f"{ctx_size[0]}x{ctx_size[1]}")
    plt.title('Średni rozmiar słownika w czasie vs Rozmiar Okna Kontekstowego')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Rozmiar Słownika')
    plt.savefig('plots/classic/context_size_analysis_dict_size.png')
    plt.cla()


    for ctx_data, ctx_size in zip(mean_1_4, context_sizes):
        sns.lineplot(x=np.arange(mean_1_4.shape[-1]), y=ctx_data[3], label = f"{ctx_size[0]}x{ctx_size[1]}")

    plt.hlines(1, xmin=0, xmax=mean_1_4.shape[-1]-1, colors='r', linestyles='dashed', label='Brak homonimów')
    plt.title('Średni wskaznik homonimów w czasie vs Rozmiar Okna Kontekstowego')
    plt.xlabel('Kroki Symulacji')
    plt.ylabel('Średni Wskaznik Homonimów')
    plt.savefig('plots/classic/context_size_analysis_homonymy_ratio.png')
    plt.cla()

baseline()
# pop_size()
# obj_size()
# mem_size()
# vocab_size()
# context_size()