import random
import numpy as np
from bigO.bigO import ab_test


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


@ab_test(lambda x: len(x), alt=insertion_sort)
def quick_sort(arr: np.ndarray) -> np.ndarray:
    sorted_arr = arr.copy()

    def _quick_sort(a, low, high):
        if low < high:
            pivot_index = partition(a, low, high)
            _quick_sort(a, low, pivot_index - 1)
            _quick_sort(a, pivot_index + 1, high)

    def partition(a, low, high):
        pivot = a[high]
        i = low - 1

        for j in range(low, high):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]

        a[i + 1], a[high] = a[high], a[i + 1]
        return i + 1

    _quick_sort(sorted_arr, 0, len(sorted_arr) - 1)
    return sorted_arr


for i in range(200):
    quick_sort(np.random.rand(random.randint(1, 100)))
