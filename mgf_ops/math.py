import numba

from math import ceil
from math import log10
from numpy.typing import NDArray


@numba.njit
def len_of_integer_part(x: float):
    if x == 0:
        return 1
    return ceil(log10(x + 1))


def test_len_of_integer_part():
    assert len_of_integer_part(0) == 1
    assert len_of_integer_part(1) == 1
    assert len_of_integer_part(2) == 1
    assert len_of_integer_part(99) == 2
    assert len_of_integer_part(100) == 3
    assert len_of_integer_part(101) == 3
    assert len_of_integer_part(999) == 3
    assert len_of_integer_part(1000) == 4
    assert len_of_integer_part(1001) == 4
    assert len_of_integer_part(9999) == 4
    assert len_of_integer_part(10000) == 5
    assert len_of_integer_part(10001) == 5


@numba.njit
def minmax(xx: NDArray) -> tuple:
    _min = xx[0]
    _max = xx[0]
    for x in xx:
        _min = min(_min, x)
        _max = max(_max, x)
    return _min, _max


@numba.njit
def count_floats(
    floats: NDArray[float],
    counts: NDArray,
    mult: int = 1000,
) -> None:
    for f in floats:
        counts[int(f * mult)] += 1
