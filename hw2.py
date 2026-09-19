from functools import lru_cache
from math import comb

@lru_cache(maxsize=None)
def d(k, n):
    if n > k:
        return 0                     
    if n == k:
        return factorial_iter(k)    
    if n == 1:
        return 2**k - 1               
    total = 0
    for j in range(1, k - (n - 1) + 1):
        total += comb(k, j) * d(k - j, n - 1)
    return total

def factorial_iter(k):
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result

print(d(15, 8))