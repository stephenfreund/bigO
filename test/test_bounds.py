import random
from bigO.check import check
import numpy as np


def square(x):
    for i in range(200):
        z = x * x
    return z


def multiply(x, n):
    z = 1
    for i in range(100):
        z += square(x)
    return x * n


def multiply_add(x, n):
    for i in range(200):
        z = multiply(x, n) + 1
    return z


@check(lambda x, y: len(x) + len(y), time_bound="O(n)")
def linear_function_2(x, y):
    for i in range(199):
        z = x + y
    return x + y


for i in range(50):
    print("combine lists 2 -- should succeed")
    linear_function_2(
        list(range(random.randint(100_000, 1_000_000))),
        list(range(random.randint(100_000, 200_000))),
    )


@check(lambda x: len(x), time_bound="O(n)", frequency=5)
def nlogn_function(x):
    return sorted(x)


for i in range(50):
    print("sort me -- should eventually fail")
    nlogn_function(list(np.random.rand(random.randint(100_000, 2_000_000))))


@check(lambda x, y: len(x) + len(y), time_bound="O(log(n))")
def linear_function_3(x, y):
    for i in range(199):
        z = x + y
    return x + y


for i in range(50):
    print("combine lists 3 -- should fail")
    linear_function_3(
        list(np.random.rand(random.randint(100_000, 1_000_000))),
        list(range(random.randint(100_000, 200_000))),
    )
