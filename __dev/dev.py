args = dict(
    in_header_format=r'TITLE=_\.(?P<clusterID_0>\d+)\.(?P<clusterID_1>\d+)\.2 File:"(?P<filename>[^"]+)", NativeID:"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+)", IonMobility:"(?P<IonMobility>[^"]+)"',
    out_header_format='TITLE=test.{clusterID_0}.{clusterID_1}.2 File:"{filename}", NativeID:"merged={merged} frame={frame} scanStart={scanStart} scanEnd={scanEnd}", IonMobility:"{IonMobility}"',
    in_mgf="outputs/regression/G8027/G8045/HYE_2024_02_16_6066entries_contaminant_tenzer/HYE_2024_02_16_6066entries_contaminant_tenzer/p12f15_1psm/p12f15_1psm/baseEdgeStats^positiveEllipse^maxRank=12_depleted/mgfs/first_gen_sage.mgf",
    fast_check="TITLE",
    verbose=True,
)

locals().update(**args)


def iter_lines(
    in_mgf: str,
    in_header_format: str,
    out_header_format: str,
    fast_check: str = "TITLE",
    verbose: bool = False,
):
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


it = iter_lines(**args)
next(it)
next(it)

# line = next(_in)
# line = next(_in)
# match == True


pattern = re.compile(
    r'TITLE=_\.(?P<clusterID_0>\d+)\.(?P<clusterID_1>\d+)\.2 File:"(?P<filename>[^"]+)", NativeID:"merged=(?P<merged>[^ ]+) frame=(?P<frame>[^ ]+) scanStart=(?P<scanStart>[^ ]+) scanEnd=(?P<scanEnd>[^ ]+)", IonMobility:"(?P<IonMobility>[^"]+)"'
)
match = pattern.search(line)
match.groupdict()
