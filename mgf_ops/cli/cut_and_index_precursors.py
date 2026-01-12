import argparse
import mmappet
import pandas as pd

from pathlib import Path


# precursors_path = "temp/F9477/correlation/transmitted_precursors.transprec/precursors.parquet"
# index_path = "temp/F9477/correlation/pmsms.mmappet/dataindex.mmappet"
# output_precursors_path = "temp/F9477/correlation/pmsms.mmappet/precursors.parquet"


def cut_precursors_and_add_indices(
    precursors_path,
    index_path,
    output_precursors_path,
) -> None:
    precursors = pd.read_parquet(precursors_path)
    idx = pd.DataFrame(mmappet.open_dataset_dct(index_path)).sort_values("ms1idx")
    final_precursors = precursors.iloc[idx.ms1idx].copy()
    final_precursors["fragment_event_cnt"] = idx["size"].to_numpy()
    final_precursors["fragment_spectrum_start"] = idx["idx"].to_numpy()
    final_precursors.to_parquet(output_precursors_path)


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
