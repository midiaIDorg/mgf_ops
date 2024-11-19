"""

TODO: This script can be simplified still with LexicographIndex.

"""
from pathlib import Path
from pprint import pprint

import click
from numba_progress import ProgressBar

import duckdb
import numba
import numpy as np
import numpy.typing as npt
import tomllib
from mmapped_df import GroupedIndex
from pandas_ops.io import read_df


@numba.njit(boundscheck=True, parallel=True)
def get_spectra_byte_counts(
    MS1_ClusterIDs,
    MS1_ClusterID_to_start_fragments,
    MS1_ClusterID_to_fragments_cnt,
    MS1_ClusterID_to_header_len,
    MS2_ClusterIDs,
    MS2_ClusterID_to_mz_intensity_len,
    _END_IONS_len,
) -> npt.NDArray:
    byte_counts = np.zeros(shape=len(MS1_ClusterID_to_header_len), dtype=np.uint64)
    for i in numba.prange(len(MS1_ClusterIDs)):
        MS1_ClusterID = MS1_ClusterIDs[i]
        fragments_start = MS1_ClusterID_to_start_fragments[MS1_ClusterID]
        fragments_end = fragments_start + MS1_ClusterID_to_fragments_cnt[MS1_ClusterID]
        byte_cnt = MS1_ClusterID_to_header_len[MS1_ClusterID]
        for _MS2_ClusterID_idx_ in range(fragments_start, fragments_end):
            MS2_ClusterID = MS2_ClusterIDs[_MS2_ClusterID_idx_]
            byte_cnt += MS2_ClusterID_to_mz_intensity_len[MS2_ClusterID]
        byte_cnt += _END_IONS_len
        byte_counts[MS1_ClusterID] = byte_cnt
    return byte_counts


# TODO: update with LexicographicIndex AT ALL COST.
def make_index(counts):
    index = np.empty(shape=(counts.shape[0] + 1,), dtype=np.uint64)
    index[0] = 0
    np.cumsum(counts, out=index[1:])
    return index


# TODO: add back m/z fragments sorting.
@numba.njit(boundscheck=True, parallel=True)
def write_spectra(
    mgf,
    MS1_ClusterIDs,
    MS1_ClusterID_to_start_fragments,
    MS1_ClusterID_to_fragments_cnt,
    MS1_ClusterID_to_header_starts,
    MS1_ClusterID_to_header_len,
    MS1_ClusterID_to_byte_idx,
    MS1_ClusterID_to_byte_cnt,
    MS1_headers,
    MS2_ClusterIDs,
    MS2_ClusterID_to_bytes,
    MS2_ClusterID_to_mz_intensity_starts,
    MS2_ClusterID_to_mz_intensity_len,
    _END_IONS_,
    progress_proxy,
) -> npt.NDArray:
    good = np.full(fill_value=False, shape=len(MS1_ClusterIDs), dtype=np.bool_)
    one = np.uint64(1)
    for i in numba.prange(len(MS1_ClusterIDs)):
        MS1_ClusterID = MS1_ClusterIDs[i]
        fragments_start = MS1_ClusterID_to_start_fragments[MS1_ClusterID]
        fragments_end = fragments_start + MS1_ClusterID_to_fragments_cnt[MS1_ClusterID]
        start_mgf_idx = mgf_idx = MS1_ClusterID_to_byte_idx[MS1_ClusterID]
        header_idx = MS1_ClusterID_to_header_starts[MS1_ClusterID]
        header_byte_cnt = MS1_ClusterID_to_header_len[MS1_ClusterID]

        for _ in range(header_byte_cnt):
            mgf[mgf_idx] = MS1_headers[header_idx]
            mgf_idx += one
            header_idx += one

        for _MS2_ClusterID_idx_ in range(fragments_start, fragments_end):
            MS2_ClusterID = MS2_ClusterIDs[_MS2_ClusterID_idx_]
            mz_intensity_idx = MS2_ClusterID_to_mz_intensity_starts[MS2_ClusterID]
            mz_intensity_byte_cnt = MS2_ClusterID_to_mz_intensity_len[MS2_ClusterID]

            for _ in range(mz_intensity_byte_cnt):
                mgf[mgf_idx] = MS2_ClusterID_to_bytes[mz_intensity_idx]
                mgf_idx += one
                mz_intensity_idx += one

        for end_ion_byte in _END_IONS_:
            mgf[mgf_idx] = end_ion_byte
            mgf_idx += one

        # check, if the total number of written bytes matches that that was anticipated.
        good[i] = np.bool_(
            (mgf_idx - start_mgf_idx) == MS1_ClusterID_to_byte_cnt[MS1_ClusterID]
        )

        progress_proxy.update(1)
    return good


@click.command(context_settings={"show_default": True})
@click.argument("precursor_cluster_stats", type=Path)
@click.argument("fragment_cluster_stats", type=Path)
@click.argument("matches", type=Path)
@click.argument("config", type=Path)
@click.argument("out_mgf", type=Path)
@click.option("--verbose", is_flag=True)
def write_mgf(
    precursor_cluster_stats: Path,
    fragment_cluster_stats: Path,
    matches: Path,
    config: Path,
    out_mgf: Path,
    verbose: bool = True,
) -> None:
    duck_con = duckdb.connect()
    with open(config, "rb") as f:
        config = tomllib.load(f)

    if verbose:
        print("Using the following MGF config:")
        pprint(config)

    edges: pd.DataFrame = read_df(matches)

    # TODO: ADD TO CONSOLIDATED CONFIG QC
    assert (
        "{precursor_stats_path}" in config["ms1"]
    ), f"The config entry `ms1` must be a query containing `{{precursor_stats_path}}` wildcard to get filled automatically with proper file system path."
    assert (
        "{fragment_stats_path}" in config["ms2"]
    ), f"The config entry `ms2` must be a query containing `{{fragment_stats_path}}` wildcard to get filled automatically with proper file system path."

    if verbose:
        print("Gathering precursor stats.")
    precursor_stats = duckdb.query(
        config["ms1"].replace("{precursor_stats_path}", str(precursor_cluster_stats))
    ).df()

    if verbose:
        print("Gathering fragment stats.")
    fragment_stats = duckdb.query(
        config["ms2"].replace("{fragment_stats_path}", str(fragment_cluster_stats))
    ).df()

    if verbose:
        print("Gathering spectra stats.")
    edges_idx = GroupedIndex(edges.MS1_ClusterID, edges)
    MS1_ClusterIDs = np.nonzero(edges_idx.counts)[0]

    _END_IONS_ = np.frombuffer(config["endions"].encode("ascii"), np.uint8)

    if verbose:
        print("Calculating how many bytes each spectrum will take.")
    MS1_ClusterID_to_byte_cnt = get_spectra_byte_counts(
        MS1_ClusterIDs=MS1_ClusterIDs,
        MS1_ClusterID_to_start_fragments=edges_idx.index,
        MS1_ClusterID_to_fragments_cnt=edges_idx.counts,
        MS1_ClusterID_to_header_len=precursor_stats.header_len.to_numpy(),
        MS2_ClusterIDs=edges.MS2_ClusterID.to_numpy(),
        MS2_ClusterID_to_mz_intensity_len=fragment_stats.mz_intensity_len.to_numpy(),
        _END_IONS_len=len(_END_IONS_),
    )

    total_bytes = np.sum(MS1_ClusterID_to_byte_cnt[MS1_ClusterIDs])
    assert total_bytes == np.sum(
        MS1_ClusterID_to_byte_cnt
    ), "Total number of bytes inconsisten with direct calculation."
    if verbose:
        print(f"The MGF will take {np.round(total_bytes / 1_000_000, 2):_} MB.")

    mgf = np.memmap(
        out_mgf,
        mode="w+",
        shape=total_bytes,
        dtype=np.uint8,
    )

    MS1_ClusterID_to_header_starts = make_index(precursor_stats.header_len.to_numpy())
    MS1_ClusterID_to_byte_idx = make_index(MS1_ClusterID_to_byte_cnt)
    MS1_ClusterID_to_bytes = np.frombuffer(
        precursor_stats.header.str.cat().encode("ascii"), dtype=np.uint8
    )
    MS2_ClusterID_to_bytes = np.frombuffer(
        fragment_stats.mz_intensity.str.cat().encode("ascii"), dtype=np.uint8
    )
    MS2_ClusterID_to_mz_intensity_starts = make_index(
        fragment_stats.mz_intensity_len.to_numpy()
    )

    if verbose:
        print("Dumping MGF to file in a hacky custom way.")

    # TODO: Can this be represented as
    with ProgressBar(
        total=len(MS1_ClusterIDs),
        desc="Dumping spectra",
        disable=not verbose,
    ) as progress:
        good = write_spectra(
            mgf=mgf,
            MS1_ClusterIDs=MS1_ClusterIDs,
            MS1_ClusterID_to_start_fragments=edges_idx.index,
            MS1_ClusterID_to_fragments_cnt=edges_idx.counts,
            MS1_ClusterID_to_header_starts=MS1_ClusterID_to_header_starts,
            MS1_ClusterID_to_header_len=precursor_stats.header_len.to_numpy(),
            MS1_ClusterID_to_byte_idx=MS1_ClusterID_to_byte_idx,
            MS1_ClusterID_to_byte_cnt=MS1_ClusterID_to_byte_cnt,
            MS1_headers=MS1_ClusterID_to_bytes,
            MS2_ClusterIDs=edges.MS2_ClusterID.to_numpy(),
            MS2_ClusterID_to_bytes=MS2_ClusterID_to_bytes,
            MS2_ClusterID_to_mz_intensity_starts=MS2_ClusterID_to_mz_intensity_starts,
            MS2_ClusterID_to_mz_intensity_len=fragment_stats.mz_intensity_len.to_numpy(),
            _END_IONS_=_END_IONS_,
            progress_proxy=progress,
        )
    assert np.all(good), "Some spectra took more bytes than anticipated."

    mgf.flush()

    if verbose:
        print("Dumped mgf.")
