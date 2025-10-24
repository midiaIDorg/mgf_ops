import numba
import numpy as np

from numpy.typing import NDArray


@numba.njit
def divide_indices(N: int, k: int = numba.get_num_threads()) -> NDArray:
    indices = np.zeros((k, 2), np.uintp)
    start = 0
    for i in range(k):
        size = (N + i) // k
        end = start + size
        indices[i, 0] = start
        indices[i, 1] = end
        start = end
    return indices


@numba.njit(parallel=True)
def count_per_batch(
    xx: NDArray,
    chunks_cnt: int = numba.get_num_threads(),
) -> NDArray:
    chunks = divide_indices(len(xx), chunks_cnt)
    max_x = xx.max()
    counts = np.zeros((chunks_cnt, max_x + 1), np.uintp)
    for i in numba.prange(chunks_cnt):
        for j in range(chunks[i, 0], chunks[i, 1]):
            counts[i, xx[j]] += 1
    return counts
