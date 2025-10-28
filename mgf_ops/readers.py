import mmappet
import pandas as pd
import pickle
import tomllib

from dictodot import DotDict
from pathlib import Path


# folder = "/home/matteo/tmp/top_prob_pseudo.msms"
def read_msms(folder: Path | str):
    folder = Path(folder)
    return DotDict(
        precursors=pd.read_parquet(folder / "precursors.parquet"),
        fragments=DotDict(
            mmappet.open_dataset_dct(folder / "fragment_spectra.mmappet")
        ),
    )


# read_msms(folder)


def parse_inputs_for_msms2mgf(
    msms_folder: Path,
    config: Path,
    mgf_path: Path,
    threads_cnt: int,
    dataset_name: str = "NA",
    verbose: bool = True,
    **kwargs
):
    res = DotDict()
    with open(config, "rb") as f:
        config = DotDict.Recursive(tomllib.load(f))
    if "{dataset}" in config.ms1_header_sql:
        config.ms1_header_sql = config.ms1_header_sql.replace("{dataset}", dataset_name)
    pseudomsms = read_msms(msms_folder)
    return DotDict(
        pseudomsms=pseudomsms,
        config=config,
        mgf_path=Path(mgf_path),
        threads_cnt=threads_cnt,
        verbose=verbose,
    )
