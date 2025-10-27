"""
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path
from pprint import pprint

import click
import duckdb
import numba
import numpy as np
import numpy.typing as npt

from dictodot import DotDict
from numba_progress import ProgressBar

from mgf_ops.readers import parse_inputs_for_msms2mgf

from mgf_ops.sortops import is_nondecreasing
from mgf_ops.sortops import strictly_increases
from mgf_ops.writers import write_spectra


DEVEL = True


@click.command(context_settings={"show_default": True})
@click.argument("msms_folder", type=Path)
@click.argument("config", type=Path)
@click.argument("mgf_path", type=Path)
@click.option("--threads_cnt", type=int, default=numba.get_num_threads())
@click.option("--dataset_name", type=str, default="NA")
@click.option("--verbose", is_flag=True)
def msms2mgf_cli(
    msms_folder: Path,
    config: Path,
    mgf_path: Path,
    threads_cnt: int,
    dataset_name: str,
    verbose: bool,
) -> None:
    msms2mgf(**parse_inputs_for_msms2mgf(**locals()))


if DEVEL:
    locals().update(
        **parse_inputs_for_msms2mgf(
            msms_folder="/home/matteo/tmp/top_prob_pseudo.msms",
            config="/home/matteo/Projects/timstofu/mgf_ops/mgf_ops/__dev/simple_mgf.toml",
            mgf_path="/home/matteo/tmp/top_prob_pseudo.mgf",
            threads_cnt=numba.get_num_threads(),
            dataset_name="B6699.d",
            verbose=True,
        )
    )
# np.nonzero(~(np.diff(pseudomsms.precursors.fragment_spectrum_start.to_numpy()) > 0))
# pseudomsms.precursors.iloc[98724 - 1 : 98724 + 2]

# from timstofu.stats import count1D

# pseudomsms.fragments.intensity.max()
# count1D

from mgf_ops.indexing import get_index
from mgf_ops.indexing import index_fragments
from mgf_ops.indexing import index_intensities
from mgf_ops.indexing import index_mzs
from mgf_ops.indexing import index_precursors

from mgf_ops.indexing import strSeries2ascii
from mgf_ops.stats import divide_indices
from mgf_ops.str_ops import ascii2str
from mgf_ops.str_ops import str2ascii
from numpy.typing import NDArray
from tqdm import tqdm

from mgf_ops.indexing import get_intensity_indexes
from mgf_ops.indexing import get_mz_indexes


import pandas as pd

from mgf_ops.math import len_of_integer_part
from mgf_ops.math import minmax
from mgf_ops.stats import count_per_batch
from mgf_ops.stats import count_per_batch


def msms2mgf(
    mgf_path: Path,
    pseudomsms: DotDict,
    config: DotDict,
    threads_cnt: int = numba.get_num_threads(),
    verbose: bool = False,
):
    if verbose:
        print("Using the following MGF config:")
        pprint(config)

    # TODO: think...
    print(f"Working with {len(pseudomsms.precursors):_} precursor peaks.")
    print(f"Working with {len(pseudomsms.fragments.mz):_} fragment peaks.")

    MZ = get_mz_indexes(pseudomsms.fragments.mz, config.fragment_mz_digits)
    INTENSITY = get_intensity_indexes(pseudomsms.fragments.intensity)

    # now, need to go fru (m/z, intensity) pairs and quantify them per precursor.
    # pseudomsms.fragments
    # int_mz_to_hash = MZ.int_mz_to_hash
    # mz_lens = MZ.str_len
    # intensity_to_hash = INTENSITY.intensity_to_hash
    # intensity_lens = INTENSITY.str_len
    # fragment_mz_digits = config.fragment_mz_digits
    # precursor_to_frag_idx = pseudomsms.precursors.fragment_spectrum_start.to_numpy()
    # precursor_to_frag_cnt = pseudomsms.precursors.fragment_event_cnt.to_numpy()

    @numba.njit(parallel=True)
    def count_ascii_per_pair(
        precursor_to_frag_idx,
        precursor_to_frag_cnt,
        fragment_mz_digits,
        mzs,
        int_mz_to_hash,
        mz_lens,
        intensities,
        intensity_to_hash,
        intensity_lens,
        additional_bytes_per_pair=2,  # " " and "\n"
        progress=None,
    ):
        precursors_cnt = len(precursor_to_frag_cnt)
        assert precursors_cnt == len(precursor_to_frag_idx)

        mz_mult = 10.0**fragment_mz_digits
        counts = np.zeros(precursors_cnt, np.uint8)
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

    # to config
    _SEPARATOR_ = str2ascii(" ")
    _NEWLINE_ = str2ascii("\n")
    _END_IONS_ = str2ascii(config["endions"])

    with ProgressBar(
        total=len(pseudomsms.precursors),
        desc="Counting ASCI lens per precursor.",
    ) as progress:
        fragments_ascii_cnts = count_ascii_per_pair(
            precursor_to_frag_idx=pseudomsms.precursors.fragment_spectrum_start.to_numpy(),
            precursor_to_frag_cnt=pseudomsms.precursors.fragment_event_cnt.to_numpy(),
            fragment_mz_digits=config.fragment_mz_digits,
            mzs=pseudomsms.fragments.mz,
            int_mz_to_hash=MZ.int_mz_to_hash,
            mz_lens=MZ.str_len,
            intensities=pseudomsms.fragments.intensity,
            intensity_to_hash=INTENSITY.intensity_to_hash,
            intensity_lens=INTENSITY.str_len,
            additional_bytes_per_pair=len(_SEPARATOR_) + len(_NEWLINE_),
            progress=progress,
        )

    headers = index_precursors(pseudomsms.precursors, config.ms1_header_sql)

    byte_cnt_per_spectrum = headers.size + fragments_ascii_cnts + len(_END_IONS_)

    total_byte_cnt = byte_cnt_per_spectrum.sum()
    print(f"MGF will take {total_byte_cnt / 1024**2:_.2f} MiB.")

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


def make_index(counts):
    index = np.empty(shape=(counts.shape[0] + 1,), dtype=np.uint64)
    index[0] = 0
    np.cumsum(counts, out=index[1:])
    return index
