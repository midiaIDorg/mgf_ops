# %load_ext autoreload
# %autoreload 2
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


if __name__ == "__main__":
    dataset_name = "F9468"
    cfg = "optimal2tier4"
    pmsms_path = Path(f"temp/{dataset_name}/{cfg}/pmsms.mmappet")
    tof2mz_path = Path(f"temp/{dataset_name}/events.ms2/tof2mz.mmappet")


@numba.njit
def count_above(xx, thr):
    cnt = 0
    for x in xx:
        if x >= thr:
            cnt += 1
    return cnt


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
