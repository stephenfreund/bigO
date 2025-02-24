import random
from bigO import bounds
import numpy as np


@bounds(lambda arr: len(arr), time="O(n*log(n))")
def insertion_sort(arr: np.ndarray) -> np.ndarray:
    sorted_arr = arr.copy()
    n = sorted_arr.shape[0]

    for i in range(1, n):
        key = sorted_arr[i]
        j = i - 1
        while j >= 0 and sorted_arr[j] > key:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key

    return sorted_arr


for i in range(100):
    insertion_sort(np.random.rand(random.randint(1, 500)))
