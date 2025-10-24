import numpy as np

from numpy.typing import NDArray


def ascii2str(ascii_arr: NDArray) -> str:
    return ascii_arr.tobytes().decode("utf-8")


def str2ascii(string: str) -> NDArray:
    return np.frombuffer(string.encode("ascii"), np.uint8)
