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


from numpy.testing import assert_equal


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

    @numba.njit(parallel=True, boundscheck=True)
    def count_ascii_per_fragment_pair(
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

    # to config
    separator = str2ascii(" ")
    newline = str2ascii("\n")
    end_ions = str2ascii(config["endions"])

    with ProgressBar(
        total=len(pseudomsms.precursors),
        desc="Counting ASCI lens per precursor.",
    ) as progress:
        fragments_ascii_cnts = count_ascii_per_fragment_pair(
            precursor_to_frag_idx=pseudomsms.precursors.fragment_spectrum_start.to_numpy(),
            precursor_to_frag_cnt=pseudomsms.precursors.fragment_event_cnt.to_numpy(),
            fragment_mz_digits=config.fragment_mz_digits,
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
    # assert_equal(np.char.str_len(headers.header.astype(str)), headers.size)

    byte_cnt_per_spectrum = headers.size + fragments_ascii_cnts + len(end_ions)
    spectrum_idx = get_index(byte_cnt_per_spectrum)

    ## slightly worrisome
    # %%time
    # obs_mzs, obs_mzs_cnts = np.unique(pseudomsms.fragments.mz, return_counts=True)
    # assert_equal(MZ.count, obs_mzs_cnts)
    # yy = count_per_batch(byte_cnt_per_spectrum).sum(0)
    # xx = yy.nonzero()[0]
    # yy = yy[xx]
    # import matplotlib.pyplot as plt
    # plt.scatter(xx, yy)
    # plt.show()

    spec_id = 0
    fragment_mz_digits = config.fragment_mz_digits

    def get_direct_spectrum(
        spec_id,
        fragment_mz_digits,
        pseudomsms,
        headers,
        end_ions,
        sep=" ",
        newlajn="\n",
    ):
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

    # from pprint import pprint

    # spectrum_direct = get_direct_spectrum(
    #     spec_id, config.fragment_mz_digits, pseudomsms, headers, config.endions
    # )
    # with open(f"/tmp/spec_{spec_id}.mgf", "w") as f:
    #     f.write(spectrum_direct)

    # pprint(spectrum_direct)
    # len(spectrum_direct)

    # spec_id = 10253

    # buggy_spec = ascii2str(mgf[spectrum_idx[spec_id] : spectrum_idx[spec_id + 1]])
    # with open(f"/tmp/buggy_spec_{spec_id}.mgf", "w") as f:
    #     f.write(buggy_spec)

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

    # would be comfy to have something like a final idx

    precursor_to_frag_idx = pseudomsms.precursors.fragment_spectrum_start.to_numpy()
    precursor_to_frag_cnt = pseudomsms.precursors.fragment_event_cnt.to_numpy()
    headers_idx = headers.idx
    headers_ascii = headers.ascii
    fragment_mz_digits = config.fragment_mz_digits
    mzs = pseudomsms.fragments.mz
    int_mz_to_hash = MZ.int_mz_to_hash
    mz_hash_to_ascii = MZ.hash_to_ascii_idx
    mz_ascii = MZ.ascii
    intensities = pseudomsms.fragments.intensity
    intensity_to_hash = INTENSITY.intensity_to_hash
    intensity_hash_to_ascii = INTENSITY.hash_to_ascii_idx
    intensity_ascii = INTENSITY.ascii
    prec_idx = 0

    @numba.njit(parallel=True, boundscheck=True)
    def fill_mgf(
        mgf,
        spectrum_idx,
        precursor_to_frag_idx,
        precursor_to_frag_cnt,
        headers_idx,
        headers_ascii,
        fragment_mz_digits,
        mzs,
        int_mz_to_hash,
        mz_hash_to_ascii,
        mz_ascii,
        intensities,
        intensity_to_hash,
        intensity_hash_to_ascii,
        intensity_ascii,
        separator,
        newline,
        end_ions,
        progress=None,
    ):
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

    # ascii2str(mgf[:mgf_idx])

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
            fragment_mz_digits=config.fragment_mz_digits,
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

    mgf.flush()

    from pprint import pprint

    pprint(ascii2str(mgf[:350]))
    pprint(ascii2str(mgf[-300:]))

    if verbose:
        print("Dumped mgf.")
