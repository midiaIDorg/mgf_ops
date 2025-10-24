import numba
import numpy as np

import numpy.typing as npt

from numba_progress import ProgressBar


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
