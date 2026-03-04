"""
End-to-end test for msms2mgf().

Run with:
    python tests/test_msms2mgf.py
"""

import tempfile
import textwrap
import numpy as np
import pandas as pd
import mmappet
from pathlib import Path

from mgf_ops.writers import msms2mgf


CONFIG_TOML = textwrap.dedent("""\
    [msms2mgf]
    ms1_header_sql = \"\"\"
    WITH numbered AS (
        SELECT
            ROW_NUMBER() OVER () AS original_idx,
            *
        FROM precursors
    )
    SELECT
        printf(
            'BEGIN IONS\\nTITLE="idx=%d frame=%d scan=%d tof=%d iim=%.4f I=%d"\\nPEPMASS=%.3f\\nRTINSECONDS=%.3f\\nCHARGE=%d+\\n',
            original_idx,
            frame,
            scan,
            tof,
            inv_ion_mobility,
            intensity,
            mz,
            rt,
            charge
        ) AS header,
        CAST(length(header) AS uinteger) AS header_len
    FROM numbered;
    \"\"\"

    [msms2mgf.fragments]
    mz_digits = 3
    mz_intensity_separator = " "
    after_intensity = "\\n"
    end_ions = "END IONS\\n\\n\\n"
""")


def create_test_pmsms(root: Path) -> tuple[Path, Path]:
    pmsms_dir = root / "pmsms.mmappet"

    # Use m/z values whose fractional parts are exact binary fractions (halves /
    # quarters / eighths) so that int(float32 * 1000) truncates correctly.
    mz = np.array(
        [100.125, 200.25,  300.5,
         150.125, 250.25,  350.5,
         120.125, 220.25,  320.5,  420.0],
        dtype=np.float32,
    )
    intensity = np.array(
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        dtype=np.uint32,
    )

    frags_df = pd.DataFrame({"mz": mz, "intensity": intensity})
    with mmappet.DatasetWriter(pmsms_dir, overwrite_dir=True) as w:
        w.append_df(frags_df)

    idx_df = pd.DataFrame({
        "ms1idx": np.array([0, 1, 2], dtype=np.int64),
        "idx":    np.array([0, 3, 6], dtype=np.int64),
        "size":   np.array([3, 3, 4], dtype=np.int64),
    })
    with mmappet.DatasetWriter(pmsms_dir / "dataindex.mmappet", overwrite_dir=True) as w:
        w.append_df(idx_df)

    precursors = pd.DataFrame({
        "fragment_spectrum_start": np.array([0, 3, 6],          dtype=np.int64),
        "fragment_event_cnt":      np.array([3, 3, 4],          dtype=np.int64),
        "frame":                   np.array([10, 20, 30],        dtype=np.int32),
        "scan":                    np.array([1,  2,  3],         dtype=np.int32),
        "tof":                     np.array([100, 200, 300],     dtype=np.int32),
        "inv_ion_mobility":        np.array([0.9, 1.0, 1.1],    dtype=np.float64),
        "intensity":               np.array([5000, 6000, 7000], dtype=np.int64),
        "mz":                      np.array([500.123, 600.456, 700.789], dtype=np.float64),
        "rt":                      np.array([10.5, 20.5, 30.5], dtype=np.float64),
        "charge":                  np.array([2, 3, 2],           dtype=np.int32),
    })
    prec_path = pmsms_dir / "filtered_precursors_with_nontrivial_ms2.parquet"
    precursors.to_parquet(prec_path)

    return pmsms_dir, prec_path


def parse_mgf(path: Path) -> list[dict]:
    spectra = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "BEGIN IONS":
                current = {"header": [], "peaks": []}
            elif line == "END IONS":
                spectra.append(current)
                current = None
            elif current is not None:
                if "=" in line:
                    current["header"].append(line)
                elif line:
                    mz_s, inten_s = line.split()
                    current["peaks"].append((float(mz_s), int(inten_s)))
    return spectra


def run_test(tmp: Path) -> None:
    pmsms_dir, prec_path = create_test_pmsms(tmp)

    config_path = tmp / "config.toml"
    config_path.write_text(CONFIG_TOML)

    out_mgf = tmp / "out.mgf"

    msms2mgf(
        pmsms_path=pmsms_dir,
        precursor_clusters_path=prec_path,
        config_path=config_path,
        out_mgf_path=out_mgf,
    )

    spectra = parse_mgf(out_mgf)

    assert len(spectra) == 3, f"Expected 3 spectra, got {len(spectra)}"

    # peak counts
    assert len(spectra[0]["peaks"]) == 3, f"Spectrum 0: expected 3 peaks, got {len(spectra[0]['peaks'])}"
    assert len(spectra[1]["peaks"]) == 3, f"Spectrum 1: expected 3 peaks, got {len(spectra[1]['peaks'])}"
    assert len(spectra[2]["peaks"]) == 4, f"Spectrum 2: expected 4 peaks, got {len(spectra[2]['peaks'])}"

    # m/z values — fractional parts are exact binary fractions so float32 truncation is safe
    assert spectra[0]["peaks"][0][0] == 100.125, f"mz mismatch: {spectra[0]['peaks'][0][0]}"
    assert spectra[1]["peaks"][0][0] == 150.125, f"mz mismatch: {spectra[1]['peaks'][0][0]}"
    assert spectra[2]["peaks"][3][0] == 420.0,   f"mz mismatch: {spectra[2]['peaks'][3][0]}"

    # intensities
    assert spectra[0]["peaks"][0][1] == 100,  f"intensity mismatch: {spectra[0]['peaks'][0][1]}"
    assert spectra[0]["peaks"][2][1] == 300,  f"intensity mismatch: {spectra[0]['peaks'][2][1]}"
    assert spectra[2]["peaks"][3][1] == 1000, f"intensity mismatch: {spectra[2]['peaks'][3][1]}"

    # header content for spectrum 0
    header_lines = {line.split("=")[0]: line for line in spectra[0]["header"]}
    assert "PEPMASS" in header_lines, "PEPMASS missing from spectrum 0 header"
    assert header_lines["PEPMASS"] == "PEPMASS=500.123", f"PEPMASS mismatch: {header_lines['PEPMASS']}"
    assert "CHARGE" in header_lines, "CHARGE missing from spectrum 0 header"
    assert header_lines["CHARGE"] == "CHARGE=2+", f"CHARGE mismatch: {header_lines['CHARGE']}"

    title_line = next((l for l in spectra[0]["header"] if l.startswith("TITLE")), None)
    assert title_line is not None, "TITLE missing from spectrum 0 header"
    assert "idx=1" in title_line,    f"TITLE missing idx=1: {title_line}"
    assert "frame=10" in title_line, f"TITLE missing frame=10: {title_line}"
    assert "scan=1" in title_line,   f"TITLE missing scan=1: {title_line}"
    assert "tof=100" in title_line,  f"TITLE missing tof=100: {title_line}"

    # charge for spectrum 1 should be 3+
    header_lines_1 = {line.split("=")[0]: line for line in spectra[1]["header"]}
    assert header_lines_1["CHARGE"] == "CHARGE=3+", f"CHARGE mismatch spectrum 1: {header_lines_1['CHARGE']}"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run_test(Path(tmp))
    print("OK")
