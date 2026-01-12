import argparse
import mmappet
import numba
import numpy as np

from mmappet.dirty_hacks_experimental_never_use import append_empty_columns
from pathlib import Path


@numba.njit
def tof2mz(tofs, tof2mz, mzs):
    assert len(tofs) == len(mzs)
    for i in range(len(tofs)):
        mzs[i] = tof2mz[tofs[i]]


# pmsms_path = Path("/home/matteo/Projects/timstofu/ionmaiden_pipeline/git/ionmaidenmetal/test_results/B6699_correlations.mmappet")
# tof2mz_path = Path("/home/matteo/Projects/timstofu/ionmaiden_pipeline/temp/B6699/fragments.ms2/tof2mz.mmappet")
# pmsms_path = Path(
#     "/home/matteo/Projects/timstofu/ionmaiden_pipeline/git/ionmaidenmetal/test_results/F9477_correlations.mmappet"
# )
# tof2mz_path = Path(
#     "/home/matteo/Projects/timstofu/ionmaiden_pipeline/temp/F9477/fragments.ms2/tof2mz.mmappet"
# )


def postprocessing_pmsms(pmsms_path, tof2mz_path):
    dataset = mmappet.open_dataset_dct(pmsms_path)
    if "mz" not in dataset:
        append_empty_columns(pmsms_path, mz=np.float32)
        dataset = mmappet.open_dataset_dct(pmsms_path, read_write=True)
        tof2mz_arr = mmappet.open_dataset_dct(tof2mz_path)["x"]
        tof2mz(dataset["tof"], tof2mz_arr, dataset["mz"])


def main():
    parser = argparse.ArgumentParser(
        description="In place add m/z values if not present."
    )
    parser.add_argument(
        "pmsms_path",
        type=Path,
        help="Path to produced pmsms.",
    )
    parser.add_argument(
        "tof2mz_path", type=Path, help="Path to (fragment) tof to mz array."
    )
    args = parser.parse_args()

    postprocessing_pmsms(args.pmsms_path, args.tof2mz_path)


if __name__ == "__main__":
    main()
