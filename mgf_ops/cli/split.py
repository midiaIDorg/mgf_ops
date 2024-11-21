import itertools
import os
from pathlib import Path

import click


def iter_spectra(file):
    current_spectrum = []
    with open(file, "r") as f:
        for line in f:
            current_spectrum.append(line)
            if "END IONS" in line:
                yield current_spectrum
                current_spectrum = []


def get_size_in_bytes(spectrum):
    return sum(len(l.encode("ascii")) for l in spectrum)


@click.command(context_settings={"show_default": True})
@click.argument("input_mgf", type=Path)
@click.argument("output_folder", type=Path)
@click.option("--max_gib_size_per_file", type=float, default=1.5)
@click.option("--createfolder", is_flag=True)
def split_mgf(
    input_mgf: Path,
    output_folder: Path,
    max_gib_size_per_file: float = 1.5,
    createfolder: bool = False,
) -> None:
    """
    Split an MGF into files smaller than a given size in GiB = 1024**3 bytes (not GB = 1000**3).

    Arguments:\n
        input_mgf (pathlib.Path): Path to the .mgf.\n
        output_folder (pathlib.Path): Path to the folder containing split mgf as separate files.\n
        max_gib_size_per_file (float): Top size of the output MGFs in GiB.
        createfolder (bool): Create folder for results. By default: not.
    """
    if createfolder:
        output_folder.mkdir(parents=True, exist_ok=True)

    assert output_folder.exists(), f"Folder `{output_folder}` does not exist."

    mgf_size_in_bytes = os.path.getsize(input_mgf)
    max_size_in_bytes = max_gib_size_per_file * 1024**3

    spectra = iter_spectra(input_mgf)
    mgf_cnt: int = 0
    spectrum: list[str] = []
    finished = False
    while not finished:
        current_mgf_size_in_bytes = 0
        with open(output_folder / f"{mgf_cnt}.mgf", "w") as out_mgf:
            current_mgf_size_in_bytes = get_size_in_bytes(spectrum)
            for line in spectrum:
                out_mgf.write(line)

            try:
                while True:
                    spectrum = next(spectra)
                    size = get_size_in_bytes(spectrum)
                    if current_mgf_size_in_bytes + size < max_size_in_bytes:
                        current_mgf_size_in_bytes += size
                        for line in spectrum:
                            out_mgf.write(line)
                    else:
                        break
            except StopIteration:
                finished = True
        mgf_cnt += 1
