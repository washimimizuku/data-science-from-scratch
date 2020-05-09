from typing import Tuple
import math

from scripts.probability.distributions import normal_cdf, inverse_normal_cdf
from scripts.hypothesis_and_inference.statistical_hypothesis_testing import (
    normal_probability_below,
    normal_probability_above
)
from scripts.hypothesis_and_inference.p_values import (
    two_sided_p_value
)

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