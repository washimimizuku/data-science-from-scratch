from typing import List

import random
import tqdm

for i in tqdm.tqdm(range(100)):
    # Do something slow
    _ = [random.random() for _ in range(100000)]

def primes_up_to(n: int) -> List[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            # i is prime if no smaller prime divides it
            i_is_prime = not any(i % p == 0 for p in primes)

            if i_is_prime:
                primes.append(i)

            t.set_description(f"{len(primes)} primes")
    
    return primes

my_primes = primes_up_to(100_000)