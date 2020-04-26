from typing import Tuple
import math
import random

# Begin previously done functions
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """Normal cumulative distribution function (PDF)"""
    return (1 + math.erf((x -mu) / math.sqrt(2) / sigma)) / 2

# The normal cdf _is_ the probability the variable is bellow a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
# End previously done functions

################################################

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

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
p_value = two_sided_p_value(529.5, mu_0, sigma_0) # 0,062

print('p_value:', p_value)

assert 0.061 < p_value < 0.063

################################################

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0    # Count # of heads
                    for _ in range(1000))                # in 1000 flips,
    if num_heads >= 530 or num_heads <= 470:             # and count how often
        extreme_value_count += 1                         # the # is 'extreme'

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

tspv = two_sided_p_value(531.5, mu_0, sigma_0) # 0.0463
assert 0.0463 < tspv < 0.0464

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_1 = upper_p_value(524.5, mu_0, sigma_0) # 0.061
assert 0.060 < upper_1 < 0.062

upper_2 = upper_p_value(526.5, mu_0, sigma_0) # 0.047
assert 0.046 < upper_2 < 0.048
