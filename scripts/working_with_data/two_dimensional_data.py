import random

import matplotlib.pyplot as plt

from scripts.working_with_data.one_dimensional_data import plot_histogram
from scripts.probability.distributions import inverse_normal_cdf
from scripts.statistics.correlation import correlation

def random_normal() -> float:
    """Returns a random draw from standard normal distribution"""
    return inverse_normal_cdf(random.random())

def main():
    xs = [random_normal() for _ in range(1000)]
    ys1 = [x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    plot_histogram(ys1, 10, "ys1 Histogram")
    plot_histogram(ys2, 10, "ys2 Histogram")

    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.title("Very Different Joint Distributions")
    plt.show()

    print(correlation(xs, ys1)) # About 0.9
    print(correlation(xs, ys2)) # About -0.9

if __name__ == "__main__": main()