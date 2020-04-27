from typing import Tuple
import math

# Begin previously done functions
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Normal cumulative distribution function (PDF)"""
    return (1 + math.erf((x -mu) / math.sqrt(2) / sigma)) / 2

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

# The normal cdf _is_ the probability the variable is bellow a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x
    (in either direction) if our values are from an N(mu, sigma)?
    """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)
# End previously done functions

################################################

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

z = a_b_test_statistic(1000, 200, 1000, 180) # -1.14
print('z:', z)
assert -1.15 < z < -1.13

p_value = two_sided_p_value(z) # 0.254
print('p_value:', p_value)
assert 0.253 < p_value < 0.255

z = a_b_test_statistic(1000, 200, 1000, 150) # -2.94
print('z:', z)
assert -2.95 < z < -2.93

p_value = two_sided_p_value(z) # 0.003
print('p_value:', p_value)
assert 0.002 < p_value < 0.004