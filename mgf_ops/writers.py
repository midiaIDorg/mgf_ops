import numba
import numpy as np

import numpy.typing as npt

from dictodot import DotDict
from numba_progress import ProgressBar
from pathlib import Path

from mgf_ops.indexing import count_ascii_per_fragment_pair
from mgf_ops.indexing import fill_mgf
from mgf_ops.indexing import get_index
from mgf_ops.indexing import get_intensity_indexes
from mgf_ops.indexing import get_mz_indexes
from mgf_ops.indexing import index_precursors

from mgf_ops.str_ops import ascii2str
from mgf_ops.str_ops import str2ascii


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
    progress_proxy: ProgressBar | None = None,
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

        if progress_proxy is not None:
            progress_proxy.update(1)

    return good


def msms2mgf(
    mgf_path: Path,
    pseudomsms: DotDict,
    config: DotDict,
    threads_cnt: int = numba.get_num_threads(),
    verbose: bool = False,
) -> None:
    if verbose:
        print("Using the following MGF config:")
        pprint(config)

    print(f"Working with {len(pseudomsms.precursors):_} precursor peaks.")
    print(f"Working with {len(pseudomsms.fragments.mz):_} fragment peaks.")

    MZ = get_mz_indexes(pseudomsms.fragments.mz, config.fragments.mz_digits)
    INTENSITY = get_intensity_indexes(pseudomsms.fragments.intensity)

    # to config
    separator = str2ascii(config.fragments.mz_intensity_separator)
    newline = str2ascii(config.fragments.after_intensity)
    end_ions = str2ascii(config.fragments.end_ions)

    with ProgressBar(
        total=len(pseudomsms.precursors),
        desc="Counting ASCI lens per precursor.",
    ) as progress:
        fragments_ascii_cnts = count_ascii_per_fragment_pair(
            precursor_to_frag_idx=pseudomsms.precursors.fragment_spectrum_start.to_numpy(),
            precursor_to_frag_cnt=pseudomsms.precursors.fragment_event_cnt.to_numpy(),
            fragment_mz_digits=config.fragments.mz_digits,
            mzs=pseudomsms.fragments.mz,
            int_mz_to_hash=MZ.int_mz_to_hash,
            mz_lens=MZ.str_len,
            intensities=pseudomsms.fragments.intensity,
            intensity_to_hash=INTENSITY.intensity_to_hash,
            intensity_lens=INTENSITY.str_len,
            additional_bytes_per_pair=len(separator) + len(newline),
            progress=progress,
        )

    headers = index_precursors(pseudomsms.precursors, config.ms1_header_sql)
    byte_cnt_per_spectrum = headers.size + fragments_ascii_cnts + len(end_ions)
    spectrum_idx = get_index(byte_cnt_per_spectrum)

    total_byte_cnt = byte_cnt_per_spectrum.sum()
    print(
        f"MGF will take {total_byte_cnt:_} bytes, or {total_byte_cnt / 1024**2:_.2f} MiB."
    )

    mgf = np.memmap(
        mgf_path,
        mode="w+",
        shape=total_byte_cnt,
        dtype=np.uint8,
    )

    with ProgressBar(
        total=len(pseudomsms.precursors),
        desc="Filling MGF with ASCII",
    ) as progress:
        assertions = fill_mgf(
            mgf=mgf,
            spectrum_idx=spectrum_idx,
            precursor_to_frag_idx=pseudomsms.precursors.fragment_spectrum_start.to_numpy(),
            precursor_to_frag_cnt=pseudomsms.precursors.fragment_event_cnt.to_numpy(),
            headers_idx=headers.idx,
            headers_ascii=headers.ascii,
            fragment_mz_digits=config.fragments.mz_digits,
            mzs=pseudomsms.fragments.mz,
            int_mz_to_hash=MZ.int_mz_to_hash,
            mz_hash_to_ascii=MZ.hash_to_ascii_idx,
            mz_ascii=MZ.ascii,
            intensities=pseudomsms.fragments.intensity,
            intensity_to_hash=INTENSITY.intensity_to_hash,
            intensity_hash_to_ascii=INTENSITY.hash_to_ascii_idx,
            intensity_ascii=INTENSITY.ascii,
            separator=separator,
            newline=newline,
            end_ions=end_ions,
            progress=progress,
        )

    assert assertions.all()
    mgf.flush()

    if verbose:
        print("Dumped mgf.")
