#!/usr/bin/env python3

import argparse
import itertools
import os
from pathlib import Path

if False:
    args = dict(
        input_mgf=Path("/home/matteo/Poligono/test.mgf"),
        out=Path("/tmp/splitmgf"),
        max_GiB_size_per_file=0.003,
    )


parser = argparse.ArgumentParser(
    description="Split an MGF into files smaller than a given size in GiB = 1024**3 bytes (not GB = 1000**3)."
)
parser.add_argument(
    "input_mgf",
    help="Path to the .mgf.",
    type=Path,
)
parser.add_argument(
    "--out",
    help="Path to the folder containing split mgf as separate files..",
    type=Path,
    required=True,
)
parser.add_argument(
    "--max_GiB_size_per_file",
    help="Top size of the output MGFs in GiB.",
    type=float,
    required=True,
)
parser.add_argument(
    "--createfolder",
    action="store_true",
    help="Drop requirement of folder's preexistance.",
)
args = parser.parse_args().__dict__


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


if __name__ == "__main__":
    if args["createfolder"]:
        args["out"].mkdir(parents=True)

    mgf_size_in_bytes = os.path.getsize(args["input_mgf"])
    max_size_in_bytes = args["max_GiB_size_per_file"] * 1024**3

    spectra = iter_spectra(args["input_mgf"])
    mgf_cnt: int = 0
    spectrum: list[str] = []
    finished = False
    while not finished:
        current_mgf_size_in_bytes = 0
        with open(args["out"] / f"{mgf_cnt}.mgf", "w") as out_mgf:
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
