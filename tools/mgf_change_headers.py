import re
import sys

if len(sys.argv) != 3:
    print("Call me maybe with:")
    print("python change_headers.py input_mgf output_mgf")
else:
    _, in_mgf, out_mgf = sys.argv

    # adjust htis if not working
    sage_title = r'TITLE=_.([^\.]+)\.([^\.]+)\.2 File:"_.d", NativeID:"merged=([^ ]+) frame=([^ ]+) scanStart=([^ ]+) scanEnd=([^ ]+)", IonMobility:"([^,]+), intensity=([^"]+)"'

    fragpipe_title = 'TITLE=test.{}.{}.2 File:"test.d", NativeID:"merged={} frame={} scanStart={} scanEnd={}", IonMobility:"{}"\n'

    sage_title_pattern = re.compile(sage_title)

    # in_mgf = "second_gen_sage.mgf"
    # out_mgf = "/tmp/fragpipe.mgf"

    with open(in_mgf, "r") as _in, open(out_mgf, "w") as _out:
        for line in _in:
            match = sage_title_pattern.search(line)
            if match:
                (
                    clusterID,
                    clusterID,
                    floor_frame_wmean,
                    floor_frame_wmean,
                    scan_min,
                    scan_max,
                    ion_mobility_wmean,
                    intensity,
                ) = match.groups()
                line = fragpipe_title.format(
                    clusterID,
                    clusterID,
                    floor_frame_wmean,
                    floor_frame_wmean,
                    scan_min,
                    scan_max,
                    ion_mobility_wmean,
                )
            _out.write(line)
