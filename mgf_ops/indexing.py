import duckdb
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numba_progress import ProgressBar
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


@numba.njit(parallel=True, boundscheck=True)
def count_ascii_per_fragment_pair(
    precursor_to_frag_idx: NDArray,
    precursor_to_frag_cnt: NDArray,
    fragment_mz_digits: int,
    mzs: NDArray,
    int_mz_to_hash: NDArray,
    mz_lens: NDArray,
    intensities: NDArray,
    intensity_to_hash: NDArray,
    intensity_lens: NDArray,
    additional_bytes_per_pair: int = 2,  # " " and "\n"
    progress: ProgressBar | None = None,
) -> NDArray:
    precursors_cnt = len(precursor_to_frag_cnt)
    assert precursors_cnt == len(precursor_to_frag_idx)

    mz_mult = 10.0**fragment_mz_digits
    counts = np.zeros(precursors_cnt, np.uintp)
    for i in numba.prange(precursors_cnt):
        s = precursor_to_frag_idx[i]
        cnt = precursor_to_frag_cnt[i]
        for frag_idx in range(s, s + cnt):
            mz = mzs[frag_idx]
            intensity = intensities[frag_idx]
            int_mz = int(mz * mz_mult)
            counts[i] += (
                mz_lens[int_mz_to_hash[int_mz]]
                + intensity_lens[intensity_to_hash[intensity]]
                + additional_bytes_per_pair
            )
        if progress is not None:
            progress.update(1)
    return counts


def get_direct_spectrum(
    spec_id: int,
    fragment_mz_digits: int,
    pseudomsms: DotDict,
    headers: DotDict,
    end_ions: str,
    sep: str = " ",
    newlajn: str = "\n",
) -> str:
    s_frag = pseudomsms.precursors.fragment_spectrum_start.iloc[spec_id]
    e_frag = s_frag + pseudomsms.precursors.fragment_event_cnt.iloc[spec_id]
    frag_mzs = pseudomsms.fragments.mz[s_frag:e_frag]
    frag_intensities = pseudomsms.fragments.intensity[s_frag:e_frag]
    str_temp = "{mz:.{fragment_mz_digits}f}{sep}{I}{newlajn}".replace(
        "{fragment_mz_digits}", str(fragment_mz_digits)
    )
    frag_tuples_repr = "".join(
        str_temp.format(mz=mz, I=I, sep=sep, newlajn=newlajn)
        for mz, I in zip(frag_mzs, frag_intensities)
    )
    header = headers.header[spec_id]
    return f"{header}{frag_tuples_repr}{end_ions}"


@numba.njit(parallel=True)
def fill_mgf(
    mgf: NDArray,
    spectrum_idx: NDArray,
    precursor_to_frag_idx: NDArray,
    precursor_to_frag_cnt: NDArray,
    headers_idx: NDArray,
    headers_ascii: NDArray,
    fragment_mz_digits: int,
    mzs: NDArray,
    int_mz_to_hash: NDArray,
    mz_hash_to_ascii: NDArray,
    mz_ascii: NDArray,
    intensities: NDArray,
    intensity_to_hash: NDArray,
    intensity_hash_to_ascii: NDArray,
    intensity_ascii: NDArray,
    separator: NDArray,
    newline: NDArray,
    end_ions: NDArray,
    progress: ProgressBar | None = None,
) -> NDArray:
    precursors_cnt = len(precursor_to_frag_cnt)
    assert precursors_cnt == len(precursor_to_frag_idx)

    mz_mult = 10.0**fragment_mz_digits
    assertions = np.empty(precursors_cnt, np.bool_)
    for prec_idx in numba.prange(precursors_cnt):
        mgf_idx = int(spectrum_idx[prec_idx])
        e_mgf = spectrum_idx[prec_idx + 1]

        # header
        s_header = headers_idx[prec_idx]
        e_header = headers_idx[prec_idx + 1]
        for header_idx in range(s_header, e_header):
            mgf[mgf_idx] = headers_ascii[header_idx]
            mgf_idx += 1

        s_frags = precursor_to_frag_idx[prec_idx]
        e_frags = s_frags + precursor_to_frag_cnt[prec_idx]
        for frag_idx in range(s_frags, e_frags):
            # f"{mz}{separator}{intensity}{newline}"
            mz_hash = int_mz_to_hash[int(mzs[frag_idx] * mz_mult)]
            s_mz = mz_hash_to_ascii[mz_hash]
            e_mz = mz_hash_to_ascii[mz_hash + 1]
            for mz_idx in range(s_mz, e_mz):
                mgf[mgf_idx] = mz_ascii[mz_idx]
                mgf_idx += 1

            for s in separator:
                mgf[mgf_idx] = s
                mgf_idx += 1

            intensity_hash = intensity_to_hash[intensities[frag_idx]]
            s_inten = intensity_hash_to_ascii[intensity_hash]
            e_inten = intensity_hash_to_ascii[intensity_hash + 1]
            for inten_idx in range(s_inten, e_inten):
                mgf[mgf_idx] = intensity_ascii[inten_idx]
                mgf_idx += 1

            for n in newline:
                mgf[mgf_idx] = n
                mgf_idx += 1

        # footer
        for e in end_ions:
            mgf[mgf_idx] = e
            mgf_idx += 1

        assertions[prec_idx] = mgf_idx == e_mgf

        if progress is not None:
            progress.update(1)

    return assertions
