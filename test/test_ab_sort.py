import random
from bigO import check
import numpy as np

from bigO.bigO import abtest, track

import numpy as np


def insertion_sort(arr: np.ndarray) -> np.ndarray:
    """
    Sorts a 1D numpy array using insertion sort, which is very fast for small arrays (< 10 elements).

    Parameters:
        arr (np.ndarray): The input array to sort.

    Returns:
        np.ndarray: A new sorted array.
    """
    # Create a copy to avoid modifying the original array.
    sorted_arr = arr.copy()
    n = sorted_arr.shape[0]

    for i in range(1, n):
        key = sorted_arr[i]
        j = i - 1
        # Shift elements that are greater than key to the right.
        while j >= 0 and sorted_arr[j] > key:
            sorted_arr[j + 1] = sorted_arr[j]
            j -= 1
        sorted_arr[j + 1] = key

    return sorted_arr


@abtest(lambda x: len(x), alt=insertion_sort)
def quick_sort(arr: np.ndarray) -> np.ndarray:
    """
    Perform quick sort on a numpy array and return a new sorted array.

    Parameters:
        arr (np.ndarray): The input 1D array to sort.

    Returns:
        np.ndarray: A sorted copy of the input array.
    """
    sorted_arr = arr.copy()  # Copy input to preserve original array

    def _quick_sort(a, low, high):
        if low < high:
            pivot_index = partition(a, low, high)
            _quick_sort(a, low, pivot_index - 1)
            _quick_sort(a, pivot_index + 1, high)

    def partition(a, low, high):
        # Using the last element as the pivot
        pivot = a[high]
        i = low - 1

        # Rearranging the elements based on the pivot value
        for j in range(low, high):
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]

        # Place the pivot in the correct position
        a[i + 1], a[high] = a[high], a[i + 1]
        return i + 1

    _quick_sort(sorted_arr, 0, len(sorted_arr) - 1)
    return sorted_arr


for i in range(200):
    print("sort me!")
    quick_sort(np.random.rand(random.randint(1, 50)))
