import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(6.5, 4))
plt.rcParams.update({
    "text.usetex": False,  # Set to True if LaTeX is installed
    "font.family": "serif",
})


def power_law():
    data = np.load('data/baseline/population_size.npy')
    y = data[:, 2].transpose(0, 2, 1).argmax(axis=2).mean(axis=1)
    population_sizes = [8, 16, 32, 64, 128, 256]
    (a, b) = np.polyfit(np.log(population_sizes), np.log(y), 1)
    print(f'Fit power law: exponent={a}, coefficient={np.exp(b)}')

    os.makedirs('plots/classic', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.loglog(population_sizes, y, label='Dane')
    plt.loglog(population_sizes, np.array(population_sizes)**(1.5), '--', label='$O(N^\gamma)$')
    plt.xlabel('Rozmiar Populacji')
    plt.ylabel('Czas do maksymalnego rozmiaru słownika (kroki)')
    plt.title('Prawo potęgowe czasu do maksymalnego rozmiaru słownika względem rozmiaru populacji')
    plt.legend()
    plt.savefig('plots/classic/power_law_population_size.png')
    plt.savefig('plots/classic/power_law_population_size.pdf')
    plt.close()

def entropy_law():
    data = np.load('data/baseline/vocab_size.npy')
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
    plt.savefig('plots/classic/entropy_law_vocab_size.pdf')
    plt.close()


power_law()
entropy_law()