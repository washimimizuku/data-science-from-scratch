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
# End previously done functions

################################################

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# The normal cdf _is_ the probability the variable is bellow a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than low
def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is not between lo and hi."""
    return 1 - normal_probability_between(lo, mu, sigma)

################################################

def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    """
    Returns the symmetric (about the mean) bounds
    that contain the specified probability
    """
    tail_probability = (1 - probability) / 2

    # Upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # Lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

print('mu_o:', mu_0)
print('sigma_0:', sigma_0)

assert mu_0 == 500
assert 15.8 < sigma_0 < 15.9

lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

print('lower_bound:', lower_bound)
print('upper_bound:', upper_bound)

assert 468.5 < lower_bound < 469.5
assert 530.5 < upper_bound < 531.5

# 95% bounds based in assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# Actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# A type 2 error means we fail to reject the null hypotesis,
# which will happen when X is still in our original interval
type_2_probabiliy = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probabiliy # 0.887

assert 0.886 < power < 0.888

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)

type_2_probabiliy = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probabiliy # 0.936

assert 0.935 < power < 0.937
