import math
import matplotlib.pyplot as plt

def uniform_pdf(x: float) -> float:
    """Uniform probability density function (PDF)"""
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    """Uniform cumulative distribution function (CDF)"""
    """Returns the probability that a uniform random variable is <= x"""
    if x < 0:
        return 0 # Uniform random is never less than 0
    elif x < 1:
        return x # e.g. P(X <= 0.4) = 0.4
    else:
        return 1 # Uniform random is always less than 1

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    """Normal probability density function (PDF)"""
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '.', label='mu=-1, sigma=1')

plt.legend()
plt.title("Various Normal pdfs")
plt.show()

def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    """Normal cumulative distribution function (PDF)"""
    return (1 + math.erf((x -mu) / math.sqrt(2) / sigma)) / 2

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '.', label='mu=-1, sigma=1')

plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()

def inverse_normal_cdf(p: float, mu: float=0, sigma: float=1, tolerance: float=0.00001) -> float:
    """Find approximate inverse using binary search"""
    # If not standart, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0                      # normal_cdf(-10) is (very close to) 0
    hi_z  =  10.0                      # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # Consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the cdf's value there
        if mid_p < p:
            low_z = mid_z              # Midpoint too low, search above it
        else:
            hi_z = mid_z               # Midpoint too high, search below it

    return mid_z

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs, [inverse_normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [inverse_normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [inverse_normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [inverse_normal_cdf(x, mu=-1) for x in xs], '.', label='mu=-1, sigma=1')

plt.legend(loc=4) # bottom right
plt.title("Various Inverse Normal cdfs")
plt.show()