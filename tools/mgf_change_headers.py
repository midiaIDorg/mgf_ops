#!/home/matteo/Projects/midia/pipelines/fishy/venv_fishy/bin/python
import argparse
import re
import sys
from pprint import pprint

import tomllib

# sage_format = r"TITLE=_\.(?P<clusterID_0>[^\.]+)\.(?P<clusterID_1>[^\.]+)\.2 File:\"_\.d\", NativeID:\"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+) IonMobility:(?P<IonMobility>[^ ]+) intensity=(?P<intensity>\w+)"

# fragpipe_format = 'TITLE=test.{clusterID_0}.{clusterID_1}.2 File:"test.d", NativeID:"merged={merged} frame={frame} scanStart={scanStart} scanEnd={scanEnd}", IonMobility:"{IonMobility}", intensity:{intensity}\n'

# args = {
#     "in_mgf": "/home/matteo/Poligono/test.mgf",
#     "out_mgf": None,
#     "in_header_format": sage_format,
#     "out_header_format": fragpipe_format,
#     "fast_check": "TITLE",
# }


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Change one flavor of MGF TITLE into another. Redirect output to STDOUT.",
)
parser.add_argument(
    "mgf",
    help="Input MGF",
)
parser.add_argument(
    "config",
    help="Configuration .toml file, containing fields: `in_header_format`, `out_header_format`, and, optionally, `fast_check`.",
)

# parser.add_argument(
#     "--in_header_format",
#     help="Header format to match to. Must use named regex groups, `?P<name>` thingies. Defaults to a flavor of midiaID SAGE.",
#     # default=r"TITLE=_\.(?P<clusterID_0>[^\.]+)\.(?P<clusterID_1>[^\.]+)\.2 File:\"_\.d\", NativeID:\"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+) IonMobility:(?P<IonMobility>[^ ]+) intensity=(?P<intensity>\w+)",
#     default=r'TITLE=_\.(?P<clusterID_0>\d+)\.(?P<clusterID_1>\d+)\.(?P<charge>\d+) File:"(?P<filename>[^"]+)", NativeID:"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+)", IonMobility:"(?P<IonMobility>[^"]+)"',
#     type=str,
# )
# parser.add_argument(
#     "--out_header_format",
#     help="Header format to match to. Must use a subset of names of groups specified in `in_header_format`. Defaults to a flavor of Fragpipe.",
#     # default='TITLE=test.{clusterID_0}.{clusterID_1}.2 File:"test.d", NativeID:"merged={merged} frame={frame} scanStart={scanStart} scanEnd={scanEnd}", IonMobility:"{IonMobility}", intensity:{intensity}\n',
#     default="TITLE={clusterID_0}.{clusterID_1}.{charge} f_{frame} SS_{scanStart} SE_{scanEnd} IM_{IonMobility}\n",
#     type=str,
# )
# parser.add_argument(
#     "--fast_check",
#     help="A faster than regex check if the line should qualify for regex match.",
#     default="TITLE",
#     type=str,
# )
args = parser.parse_args().__dict__


def iter_lines(
    in_mgf: str,
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


if __name__ == "__main__":
    with open(args["config"], "rb") as f:
        config = tomllib.load(f)
    assert config[
        "in_header_format"
    ], f"Missing `in_header_format` in {args['config']}."
    assert config[
        "out_header_format"
    ], f"Missing `out_header_format` in {args['config']}."
    if "fast_check" not in config:
        config["fast_check"] = ""
    config["in_mgf"] = args["mgf"]
    for line in iter_lines(**config):
        print(line, end="")
