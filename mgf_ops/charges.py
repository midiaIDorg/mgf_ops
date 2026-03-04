import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def encode_charges_bool(
    precursor_charge_matrix: NDArray,
    labels: NDArray,
    zeroisall: bool = False,
) -> NDArray:
    n, m = precursor_charge_matrix.shape
    out = np.empty(n, dtype=np.int64)

    # Precompute "all labels" code once if needed
    all_code = 0
    if zeroisall:
        for j in range(m):
            all_code = all_code * 10 + labels[j]

    for i in range(n):
        acc = 0
        found = False

        for j in range(m):
            if precursor_charge_matrix[i, j]:
                acc = acc * 10 + labels[j]
                found = True

        if not found and zeroisall:
            acc = all_code

        out[i] = acc

    return out


@numba.njit
def digits_base10(n: int) -> NDArray:
    n = abs(n)

    # Special case
    if n == 0:
        out = np.empty(1, dtype=np.int64)
        out[0] = 0
        return out

    # First pass: count digits
    tmp = n
    count = 0
    while tmp > 0:
        count += 1
        tmp //= 10

    # Allocate output
    out = np.empty(count, dtype=np.int64)

    # Second pass: fill digits from the end
    i = count - 1
    while n > 0:
        out[i] = n % 10
        n //= 10
        i -= 1

    return out


@numba.njit
def explode_charge_codes(charge_codes: NDArray) -> tuple[list[int], list[int]]:
    ids = []
    charges = []
    for i, charge_code in enumerate(charge_codes):
        for q in digits_base10(charge_code):
            ids.append(i)
            charges.append(q)
    return ids, charges
