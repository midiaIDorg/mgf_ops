import duckdb
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numpy.testing import assert_equal
from numpy.typing import NDArray
from typing import Any


from mgf_ops.math import count_floats
from mgf_ops.math import minmax
from mgf_ops.stats import count_per_batch


@numba.njit(boundscheck=True)
def get_index(counts: NDArray, index: NDArray | None = None) -> NDArray:
    """Turn counts into cumulated sums offset by one 0 at the beginning.

    This function should be used to create an indexed view of elements in another table where elements are in groups and counts summarizes how many times they occur in those groups.
    See test_get_index.

    Arguments:
        counts (NDArray): An array of counts.
        index (NDArray|None): Optional place to store index.

    Returns:
        np.array: A table with 0 and then cumulated counts.
    """
    if index is None:
        index = np.empty(shape=len(counts) + 1, dtype=np.int64)
    assert len(index) == len(counts) + 1
    index[0] = 0
    i = 1
    for cnt in counts:
        index[i] = index[i - 1] + cnt
        i += 1
    return index


def strSeries2ascii(xx: pd.Series | NDArray) -> NDArray:
    return np.frombuffer(
        pd.Series(xx, copy=False).str.cat().encode("ascii"), dtype=np.uint8
    )


def index_mzs(
    mzs: NDArray,
    digits: int = 4,
    duck_con: duckdb.DuckDBPyConnection | None = None,
) -> DotDict[str, NDArray]:
    """Make a big array of uint8s containing all m/zs written down as ascii characters and concatenated. Provided index allows to retrieve the representation.

    We do it because of the mgfs. They should have never existed. #MGFsucks
    """
    if duck_con is None:
        duck_con = duckdb.connect()
    duck_con.register(
        "mzs",
        pd.DataFrame(
            dict(mz=mzs),
            copy=False,
        ),
    )
    sql = """
    SELECT
    printf('%.{digits}f',mz) AS mz_str,
    CAST(length(mz_str) AS uinteger) AS mz_str_len
    FROM 'mzs'
    """
    sql = sql.replace("{digits}", str(digits))
    possible_mzs = duck_con.query(sql).df()
    return DotDict(
        idx=get_index(possible_mzs.mz_str_len.to_numpy()),
        ascii=strSeries2ascii(possible_mzs.mz_str),
    )


def index_intensities(
    intensities: NDArray,
    duck_con: duckdb.DuckDBPyConnection | None = None,
) -> DotDict[str, NDArray]:
    """Make a big array of uint8s containing all intensities written down as ascii characters and concatenated. Provided index allows to retrieve the representation of a given intensity.

    We do it because of the mgfs. They should have never existed. #MGFsucks
    """
    if duck_con is None:
        duck_con = duckdb.connect()
    duck_con.register(
        "intensities",
        pd.DataFrame(
            dict(intensity=np.arange(len(intensities))),
            copy=False,
        ),
    )
    possible_intensities = duck_con.query(
        """
    SELECT
    CAST(intensity AS TEXT) AS intensity_str,
    CAST(length(intensity_str) AS uinteger) AS intensity_str_len
    FROM 'intensities'
    """
    ).df()
    return DotDict(
        idx=get_index(possible_intensities.intensity_str_len.to_numpy()),
        ascii=strSeries2ascii(possible_intensities.intensity_str),
    )


def index_precursors(
    precursors: pd.DataFrame,
    ms1_header_sql: str,
    duck_con: duckdb.DuckDBPyConnection | None = None,
) -> DotDict[str, NDArray]:
    if duck_con is None:
        duck_con = duckdb.connect()
    duck_con.register("precursors", precursors)
    precursor_stats = duck_con.query(ms1_header_sql).df()
    return DotDict(
        size=precursor_stats.header_len.to_numpy(),
        idx=get_index(precursor_stats.header_len.to_numpy()),
        ascii=strSeries2ascii(precursor_stats.header),
        header=precursor_stats.header.to_numpy(),
    )


def index_fragments(
    precursors: pd.DataFrame,
    fragments: DotDict[str, NDArray],
    tof2mz_arr: NDArray,
    fragment_mz_digits: int = 4,
    paranoid: bool = False,  # but almost...
) -> tuple[DotDict[str, DotDict[str, NDArray]], DotDict]:
    assert (
        "fragment_spectrum_start" in precursors.columns
    ), "Precursors need to contain information where they start."

    meta = DotDict(
        {f"{c}_counts_per_batch": count_per_batch(v) for c, v in fragments.items()}
    )
    for c in fragments:
        meta[c] = meta[f"{c}_counts_per_batch"].sum(0)
    meta.mz_intensity_pair_cnt_per_batch = meta.intensity_counts_per_batch.sum(1)
    assert_equal(
        meta.mz_intensity_pair_cnt_per_batch,
        meta.tof_counts_per_batch.sum(1),
        err_msg="The number of tofs and intensities differ per batch.",
    )
    index = DotDict(
        mzs=index_mzs(tof2mz_arr[np.arange(len(meta.tof))].round(fragment_mz_digits)),
        intensities=index_intensities(meta.intensity),
    )

    if paranoid:
        for _tof in tqdm(
            np.random.choice(fragments.tof.max(), size=10_000, replace=False),
            desc="Checking if we represent m/z-s correctly",
        ):
            assert abs(
                tof2mz_arr[_tof].round(fragment_mz_digits)
                - float(
                    ascii2str(
                        meta.fragment.mzs.ascii[
                            meta.fragment.mzs.idx[_tof] : meta.fragment.mzs.idx[
                                _tof + 1
                            ]
                        ]
                    )
                )
            ) < 10.0 ** (-(fragment_mz_digits - 1))
    return index, meta


def get_mz_indexes(
    mzs: NDArray[float],
    mz_digits: int = 3,
    duck_con: duckdb.DuckDBPyConnection | None = None,
) -> DotDict[str, Any]:
    R = DotDict()
    R.min_mz, R.max_mz = minmax(mzs)
    assert isinstance(mz_digits, int)
    assert mz_digits > 0
    mult = 10.0**mz_digits

    int_mz_counts = np.zeros(int(R.max_mz * mult) + 1, np.uint32)
    count_floats(mzs, int_mz_counts, mult)

    R.int_mz = int_mz_counts.nonzero()[0]
    R.count = int_mz_counts[R.int_mz]

    if duck_con is None:
        duck_con = duckdb.connect()

    duck_con.register("X", pd.DataFrame(dict(int_mz=R.int_mz), copy=False))

    sql = f"""
    SELECT
    printf('%.{mz_digits}f', int_mz / {mult}) AS str,
    CAST(length(str) AS uinteger) AS str_len
    FROM 'X'
    """
    for c, v in duck_con.query(sql).df().items():
        R[c] = v.to_numpy()

    R.hash_to_ascii_idx = get_index(R.str_len)
    R.ascii = strSeries2ascii(R.str)
    R.int_mz_to_hash = int_mz_counts  # reusing space
    R.int_mz_to_hash[R.int_mz] = np.arange(len(R.int_mz))

    return R


def test_get_mz_indexes():
    mzs = np.array(
        [100.01, 100.02, 100.003, 100.002, 101.122, 1020.2132, 0.0, 0.121, 10.12]
    )
    mz_digits = 2
    mult = 10.0**2
    MZ = get_mz_indexes(mzs, mz_digits)
    assert MZ.min_mz == mzs.min()
    assert MZ.max_mz == mzs.max()
    unique_sorted_mzs, mz_counts = np.unique(
        (np.sort(mzs) * mult).round().astype(int), return_counts=True
    )
    assert_equal(MZ.int_mz, unique_sorted_mzs)
    unique_sorted_mzs_str = np.array(
        [f"{m / mult:.2f}" for m in unique_sorted_mzs]
    ).astype(str)
    assert_equal(MZ.str.astype(str), unique_sorted_mzs_str)
    assert_equal(np.char.str_len(MZ.str.astype(str)), MZ.str_len)
    for i in range(len(unique_sorted_mzs_str)):
        mz_str = unique_sorted_mzs_str[i]
        _ascii = MZ.ascii[MZ.hash_to_ascii_idx[i] : MZ.hash_to_ascii_idx[i + 1]]
        assert ascii2str(_ascii) == mz_str


def get_intensity_indexes(intensities: NDArray[int]) -> DotDict[str, Any]:
    intensity_cnts = count_per_batch(intensities).sum(0)
    R = DotDict(observed=intensity_cnts.nonzero()[0])
    intensity_cnts[R.observed] = np.arange(len(R.observed))
    R.intensity_to_hash = intensity_cnts
    R.str = R.observed.astype(str)
    R.str_len = np.char.str_len(R.str)
    R.hash_to_ascii_idx = get_index(R.str_len)
    R.ascii = strSeries2ascii(R.str)
    return R


def test_get_intensity_indexes():
    intensities = np.array([1, 0, 120, 24, 2432, 3244, 242, 10])
    I = get_intensity_indexes(intensities)
    intensities_sorted = np.sort(intensities)
    assert_equal(intensities_sorted, I.observed)
    intensities_str = intensities_sorted.astype(str)
    assert_equal(intensities_str, I.str)
    assert_equal(np.char.str_len(intensities_str), np.char.str_len(I.str))
    for i in range(len(I.str)):
        assert I.str[i] == ascii2str(
            I.ascii[I.hash_to_ascii_idx[i] : I.hash_to_ascii_idx[i + 1]]
        )
