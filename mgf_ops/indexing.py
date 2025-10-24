import duckdb
import numba
import numpy as np
import pandas as pd

from dictodot import DotDict
from numpy.typing import NDArray

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
        index = np.empty(shape=len(counts) + 1, dtype=np.uint32)
    assert len(index) == len(counts) + 1
    index[0] = 0
    i = 1
    for cnt in counts:
        index[i] = index[i - 1] + cnt
        i += 1
    return index


def strSeries2ascii(xx: pd.Series) -> NDArray:
    return np.frombuffer(xx.str.cat().encode("ascii"), dtype=np.uint8)


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
        idx=get_index(precursor_stats.header_len.to_numpy()),
        ascii=strSeries2ascii(precursor_stats.header),
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
    np.testing.assert_equal(
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
