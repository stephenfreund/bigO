import random
from bigO import check
import numpy as np

from bigO.bigO import abtest


def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


@abtest(lambda x: len(x), alt=selection_sort)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


@abtest(lambda x: len(x), alt=quick_sort)
def quick_sort2(arr):
    """
    A version of qsort done with loops
    """ 
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = []
    middle = []
    right = []
    for x in arr:
        if x < pivot:
            left.append(x)
        elif x == pivot:
            middle.append(x)
        else:
            right.append(x)
    return quick_sort(left) + middle + quick_sort(right)


for i in range(10):
    print("sort me!")
    quick_sort(list(np.random.rand(random.randint(100, 1000))))
    quick_sort2(list(np.random.rand(random.randint(100, 50000))))
