from typing import Tuple
import math
import random

from scripts.probability.distributions import normal_cdf, inverse_normal_cdf
from scripts.hypothesis_and_inference.statistical_hypothesis_testing import (
    normal_probability_below,
    normal_probability_above,
    normal_approximation_to_binomial,
    normal_upper_bound,
    normal_lower_bound,
    normal_two_sided_bounds
)

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

def main():
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

    p_hat = 525 / 1000
    mu = p_hat
    sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
    assert 0.0157 < sigma < 0.0159

    two_sided_bounds = normal_two_sided_bounds(0.95, mu, sigma) # [0.4940, 0.5560]
    assert 0.4939 < two_sided_bounds[0] < 0.4941
    assert 0.5559 < two_sided_bounds[1] < 0.5561

    p_hat = 540 / 1000
    mu = p_hat
    sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
    assert 0.0157 < sigma < 0.0159

    two_sided_bounds = normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709]
    assert 0.5090 < two_sided_bounds[0] < 0.5092
    assert 0.5708 < two_sided_bounds[1] < 0.5710

if __name__ == "__main__": main()