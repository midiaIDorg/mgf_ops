"""
%load_ext autoreload
%autoreload 2
"""
from pathlib import Path
from pprint import pprint

import click
import numba
import numpy as np

from dictodot import DotDict
from numba_progress import ProgressBar

from mgf_ops.indexing import count_ascii_per_fragment_pair
from mgf_ops.indexing import fill_mgf
from mgf_ops.indexing import get_index
from mgf_ops.indexing import get_intensity_indexes
from mgf_ops.indexing import get_mz_indexes
from mgf_ops.indexing import index_precursors
from mgf_ops.readers import parse_inputs_for_msms2mgf
from mgf_ops.str_ops import ascii2str
from mgf_ops.str_ops import str2ascii

DEVEL = False


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

    assert assertions.all()
    mgf.flush()

    if verbose:
        print("Dumped mgf.")


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


# would be comfy to have something like a final idx
# precursor_to_frag_idx = pseudomsms.precursors.fragment_spectrum_start.to_numpy()
# precursor_to_frag_cnt = pseudomsms.precursors.fragment_event_cnt.to_numpy()
# headers_idx = headers.idx
# headers_ascii = headers.ascii
# fragment_mz_digits = config.fragment_mz_digits
# mzs = pseudomsms.fragments.mz
# int_mz_to_hash = MZ.int_mz_to_hash
# mz_hash_to_ascii = MZ.hash_to_ascii_idx
# mz_ascii = MZ.ascii
# intensities = pseudomsms.fragments.intensity
# intensity_to_hash = INTENSITY.intensity_to_hash
# intensity_hash_to_ascii = INTENSITY.hash_to_ascii_idx
# intensity_ascii = INTENSITY.ascii
# prec_idx = 0
