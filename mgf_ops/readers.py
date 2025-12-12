import mmappet
import pandas as pd
import pickle
import tomllib

from dictodot import DotDict
from pathlib import Path


def open_pmsms(directory: str | Path) -> DotDict:
    directory = Path(directory)
    return DotDict(
        fragments=DotDict(mmappet.open_dataset_dct(directory)),
        idx=DotDict(mmappet.open_dataset_dct(directory / "dataindex.mmappet")),
        precursors=pd.read_parquet(directory / "precursors.parquet"),
    )


## TODO: update the old path output structure.
# def read_msms(folder: Path | str):
#     folder = Path(folder)
#     return DotDict(
#         precursors=pd.read_parquet(folder / "precursors.parquet"),
#         fragments=DotDict(
#             mmappet.open_dataset_dct(folder / "fragment_spectra.mmappet")
#         ),
#     )


def parse_inputs_for_msms2mgf(
    msms_folder: Path,
    config: Path,
    mgf_path: Path,
    threads_cnt: int,
    verbose: bool = True,
    **kwargs
):
    res = DotDict()
    with open(config, "rb") as f:
        config = DotDict.Recursive(tomllib.load(f)["msms2mgf"])
    pseudomsms = open_pmsms(msms_folder)
    if "idx" in pseudomsms:
        pseudomsms.precursors["fragment_event_cnt"] = pseudomsms.idx.size
        pseudomsms.precursors["fragment_spectrum_start"] = pseudomsms.idx.idx
        pseudomsms.precursors = pseudomsms.precursors.iloc[pseudomsms.idx.ms1idx]

    return DotDict(
        pseudomsms=pseudomsms,
        config=config,
        mgf_path=Path(mgf_path),
        threads_cnt=threads_cnt,
        verbose=verbose,
    )


if __name__ == "__main__":
    msms_folder = Path(
        "/home/matteo/Projects/timstofu/ionmaiden_pipeline/temp/F9477/multivoxelrefit/pmsms.mmappet"
    )
    config = Path(
        "/home/matteo/Projects/timstofu/ionmaiden_pipeline/configs/multivoxelrefit.toml"
    )
    mgf_path = Path(
        "/home/matteo/Projects/timstofu/ionmaiden_pipeline/temp/F9477/multivoxelrefit/mgf.mgf"
    )
    threads_cnt = 16
    verbose = True
