#!/usr/bin/env python
import argparse
import re
import sys

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
    description="Change one flavor of MGF headers into another.",
)
parser.add_argument(
    "in_mgf",
    help="Input MGF",
)
parser.add_argument(
    "--in_header_format",
    help="Header format to match to. Must use named regex groups, `?P<name>` thingies. Defaults to a flavor of midiaID SAGE.",
    default=r"TITLE=_\.(?P<clusterID_0>[^\.]+)\.(?P<clusterID_1>[^\.]+)\.2 File:\"_\.d\", NativeID:\"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+) IonMobility:(?P<IonMobility>[^ ]+) intensity=(?P<intensity>\w+)",
    type=str,
)
parser.add_argument(
    "--out_header_format",
    help="Header format to match to. Must use a subset of names of groups specified in `in_header_format`. Defaults to a flavor of Fragpipe.",
    default='TITLE=test.{clusterID_0}.{clusterID_1}.2 File:"test.d", NativeID:"merged={merged} frame={frame} scanStart={scanStart} scanEnd={scanEnd}", IonMobility:"{IonMobility}", intensity:{intensity}\n',
    type=str,
)
parser.add_argument(
    "--fast_check",
    help="A faster than regex check if the line should qualify for regex match.",
    default="TITLE",
    type=str,
)
args = parser.parse_args().__dict__


if __name__ == "__main__":
    in_header_pattern = re.compile(args["in_header_format"])
    with open(args["in_mgf"], "r") as _in:
        for line in _in:
            if (
                len(args["fast_check"]) == 0
                or line[: len(args["fast_check"])] == args["fast_check"]
            ):
                match = in_header_pattern.search(line)
                if match:
                    line = args["out_header_format"].format(**match.groupdict())
            print(line, end="")
