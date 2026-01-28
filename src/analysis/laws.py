import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


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


