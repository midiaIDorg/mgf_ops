import click
import numba

from pathlib import Path

from mgf_ops.readers import parse_inputs_for_msms2mgf
from mgf_ops.writers import msms2mgf


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
