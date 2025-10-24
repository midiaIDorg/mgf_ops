import math
import numba

from numpy.typing import NDArray


@numba.njit
def strictly_increases(xx: NDArray):
    x_prev = -math.inf
    for x in xx:
        if x_prev >= x:
            return False
        x_prev = x
    return True


@numba.njit(cache=True)
def is_nondecreasing(xx):
    x_prev = -math.inf
    for x in xx:
        if x_prev > x:
            return False
        x_prev = x
    return True
