import re
from pathlib import Path

import click

import tomllib


def iter_lines(
    in_mgf: Path | str,
    in_header_format: str,
    out_header_format: str,
    fast_check: str = "",
    verbose: bool = False,
    **kwargs,
):
    """
    Iter line after a line and pass it through if it does not match fast check. Otherwise try to match the pattern.
    """
    in_header_pattern = re.compile(in_header_format)
    with open(in_mgf, "r") as _in:
        for line in _in:
            if len(fast_check) == 0 or line[: len(fast_check)] == fast_check:
                if verbose and len(fast_check) > 0:
                    print("fast_check matched")
                match = in_header_pattern.search(line)
                if match:
                    old_line = line
                    line = out_header_format.format(**match.groupdict())
                    if verbose:
                        print(f"Changed:\n{old_line}\nto\n{line}")
            yield line


@click.command(context_settings={"show_default": True})
@click.argument("in_mgf", type=Path)
@click.argument("config", type=Path)
@click.option("--verbose", is_flag=True)
def change_mgf_headers(
    in_mgf: Path,
    config: Path,
    verbose: bool = False,
) -> None:
    """ "Change one flavor of MGF TITLE into another. Redirect output to STDOUT."

    Arguments:
        in_mgf (pathlib.Path): The input MGF for which one want to apply the changes.
        config (pathlib.Path): Configuration .toml file, containing fields: `in_header_format`, `out_header_format`, and, optionally, `fast_check`.

    """
    with open(config, "rb") as f:
        conf = tomllib.load(f)

    for requested_key in ("in_header_format", "out_header_format"):
        assert conf[requested_key], f"Missing `{requested_key}` in {conf}."

    for line in iter_lines(
        in_mgf=in_mgf,
        in_header_format=conf["in_header_format"],
        out_header_format=conf["out_header_format"],
        fast_check=conf.get("fast_check", ""),
        verbose=verbose,
    ):
        print(line.strip())
