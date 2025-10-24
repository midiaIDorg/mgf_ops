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

from mgf_ops.indexing import index_fragments
from mgf_ops.indexing import index_intensities
from mgf_ops.indexing import index_mzs
from mgf_ops.indexing import index_precursors

from mgf_ops.str_ops import ascii2str
from mgf_ops.str_ops import str2ascii
from tqdm import tqdm


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

    # TODO: how the heck can this be divided into chunks if I need to divide that with precursors???? fix that chunking.

    # INSTEAD: USE THIS BELOW AND SPLIT THAT AND THEN COUNT ON THESE CHUNKS....,
    # OR, BETTER, USE THIS CHUNKING AND FIND OUT WHICH PRECURSORS ARE INVOLVED.

    # fragment_index = precursors[
    #     ["fragment_spectrum_start", "fragment_spectrum_end"]
    # ].to_numpy()
    fragments, fragmeta = index_fragments(
        pseudomsms.precursors,
        pseudomsms.fragments,
        pseudomsms.meta.tof2mz_arr,
        config.fragment_mz_digits,
    )
    fragmeta.mz_intensity_pair_cnt_per_batch
    headers = index_precursors(pseudomsms.precursors, config.ms1_header_sql)

    k = 1000
    ascii2str(headers.ascii[headers.idx[k] : headers.idx[k + 1]])
    _END_IONS_ = str2ascii(config["endions"])

    if verbose:
        print("Calculating how many bytes each spectrum will take per thread.")

    numba.set_num_threads(threads_cnt)
    if verbose:
        print("Gathering fragment stats.")

    # fragment_stats = duckdb.query(
    #     conf["ms2"].replace("{fragment_stats_path}", str(fragment_cluster_stats))
    # ).df()

    spectra_cnt = len(pseudomsms.precursors)

    @numba.njit(boundscheck=True, parallel=True)
    def get_spectra_byte_counts(
        spectra_cnt,
        MS1_ClusterID_to_start_fragments,
        MS1_ClusterID_to_fragments_cnt,
        MS1_ClusterID_to_header_len,
        MS2_ClusterIDs,
        MS2_ClusterID_to_mz_intensity_len,
        _END_IONS_len,
    ) -> npt.NDArray:
        byte_counts = np.zeros(shape=spectra_cnt, dtype=np.uint64)

        for i in numba.prange(len(MS1_ClusterIDs)):
            MS1_ClusterID = MS1_ClusterIDs[i]
            fragments_start = MS1_ClusterID_to_start_fragments[MS1_ClusterID]
            fragments_end = (
                fragments_start + MS1_ClusterID_to_fragments_cnt[MS1_ClusterID]
            )
            byte_cnt = MS1_ClusterID_to_header_len[MS1_ClusterID]
            for _MS2_ClusterID_idx_ in range(fragments_start, fragments_end):
                MS2_ClusterID = MS2_ClusterIDs[_MS2_ClusterID_idx_]
                byte_cnt += MS2_ClusterID_to_mz_intensity_len[MS2_ClusterID]
            byte_cnt += _END_IONS_len
            byte_counts[MS1_ClusterID] = byte_cnt
        return byte_counts

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


def make_index(counts):
    index = np.empty(shape=(counts.shape[0] + 1,), dtype=np.uint64)
    index[0] = 0
    np.cumsum(counts, out=index[1:])
    return index
