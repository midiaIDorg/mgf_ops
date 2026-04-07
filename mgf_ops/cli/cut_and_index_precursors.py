# %load_ext autoreload
# %autoreload 2
import argparse
import mmappet
import numpy as np
import pandas as pd

from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)


if __name__ == "__main__":
    dataset_name = "F9468"
    dataset_name = "G20513"
    cfg = "optimal2tier4"
    cfg = "optimal2tier"
    precursors_path = Path(
        f"temp/{dataset_name}/{cfg}/filtered_transmitted_precursor_clusters.parquet"
    )
    index_path = Path(f"temp/{dataset_name}/{cfg}/pmsms.mmappet/dataindex.mmappet")
    output_precursors_path = (
        "/home/matteo/tmp/{dataset_name}/{cfg}/pmsms.mmappet/precursors.parquet"
    )


def cut_precursors_and_add_indices(
    precursors_path,
    index_path,
    output_precursors_path,
) -> None:
    precursors = pd.read_parquet(precursors_path)
    idx_all = pd.DataFrame(mmappet.open_dataset_dct(index_path), copy=False)
    idx = (
        idx_all
        .sort_values("ms1idx")
        .query("size > 0")
        .sort_values("idx")
    )
    n_empty = (idx_all["size"] == 0).sum()
    if n_empty:
        print(f"Dropped {n_empty:_} / {len(idx_all):_} precursors with no fragment events (size == 0 in mkpmsms output).")

    precursors = precursors.sort_values("transmitted_idx").reset_index(drop=True)
    final_precursors = precursors.iloc[
        idx.ms1idx
    ].copy()  # sorts precursors by reported spectra
    final_precursors["fragment_event_cnt"] = idx["size"].to_numpy()
    final_precursors["fragment_spectrum_start"] = idx["idx"].to_numpy()
    assert np.all(np.diff(final_precursors.fragment_spectrum_start) > 0)
    final_precursors.to_parquet(output_precursors_path, index=False)
    print("Filtered Precursors with Nontrivial MS2 Spectra:")
    print(final_precursors)


def main():
    parser = argparse.ArgumentParser(description="Cut precursors and add indices.")
    parser.add_argument(
        "precursors_path",
        type=Path,
        help="Path to precursors chosen for pmsms (transmitted ones).",
    )
    parser.add_argument("index_path", type=Path, help="Path where the index is saved.")
    parser.add_argument(
        "output_precursors_path",
        type=Path,
        help="Path to the precursors.",
    )
    args = parser.parse_args()

    cut_precursors_and_add_indices(
        args.precursors_path, args.index_path, args.output_precursors_path
    )


if __name__ == "__main__":
    main()
