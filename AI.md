# CLAUDE.md — mgfops

## Purpose

`mgf_ops` is a Python library for converting mass spectrometry pseudo-MSMS data into
MGF (Mascot Generic Format) files. It prioritizes throughput: m/z and intensity values
are hashed once to pre-computed ASCII byte arrays, total byte counts are computed upfront
to allocate an exact-size `np.memmap`, and all hot loops run under Numba with `prange`.

---

## Pipeline position

The typical data flow before `msms2mgf` runs:

1. **`postprocess_pmsms`** — adds `mz` (float32) to a raw `.mmappet` dataset by
   converting TOF indices via a lookup table (`tof2mz.mmappet`). Operates in-place.

2. **`cut_and_index_precursors`** — reads a precursors parquet and a
   `dataindex.mmappet`, filters to precursors with non-trivial MS2, and writes
   `filtered_precursors_with_nontrivial_ms2.parquet` with columns
   `fragment_event_cnt` and `fragment_spectrum_start` added.

3. **`msms2mgf`** — reads the pmsms folder produced above and writes the MGF file.

---

## pmsms folder layout (Format 2 — primary)

`open_pmsms(directory)` in `readers.py` opens a `.mmappet` folder as:

```
pmsms.mmappet/
  filtered_precursors_with_nontrivial_ms2.parquet   ← precursor table
  mz        (float array)                            ← fragment m/z values
  intensity  (int array)                             ← fragment intensities
  dataindex.mmappet/                                 ← index sub-folder
```

Returns a `DotDict` with keys `fragments`, `idx`, `precursors`.

Required precursor columns: `fragment_spectrum_start` (int), `fragment_event_cnt` (int).

---

## Package layout

| File / folder | Role |
|---|---|
| `readers.py` | `open_pmsms()`, `parse_inputs_for_msms2mgf()` |
| `indexing.py` | Core hashing/indexing: `get_mz_indexes()`, `get_intensity_indexes()`, `count_ascii_per_fragment_pair()`, `fill_mgf()`, `index_precursors()`, `get_index()` |
| `writers.py` | `msms2mgf()`, `msms2mgf_multicharge()`, `expand_precursors_by_charges()`, `write_spectra()` Numba kernel (legacy) |
| `charges.py` | Numba helpers for charge encoding/decoding: `encode_charges_bool()`, `digits_base10()`, `explode_charge_codes()` |
| `math.py` | Numba utilities: `count_floats()`, `minmax()`, `len_of_integer_part()` |
| `stats.py` | `count_per_batch()`, `divide_indices()` — parallel histogram counting |
| `sortops.py` | Sorting helpers |
| `str_ops.py` | `ascii2str()`, `str2ascii()` |
| `cli/` | Click / argparse entry points (one file per command) |
| `configs/` | Bundled TOML configs (`sage2peaks_*`) for `change_mgf_headers` |
| `__dev/` | Exploratory scripts — not installed |

---

## CLI entry points

Declared in `pyproject.toml [project.scripts]`:

| Command | File | Status |
|---|---|---|
| `msms_one_charge` | `mgf_ops.writers:msms_one_charge_cli` | Working — single `charge` column per precursor |
| `msms_multiple_charge` | `mgf_ops.writers:msms_multiple_charge_cli` | Working — `charges` int column, one entry per encoded digit |
| `write_mgf` | `mgf_ops.cli.write:write_mgf` | **Broken** — uses commented-out `GroupedIndex`/`read_df` imports |
| `split_mgf` | `mgf_ops.cli.split:split_mgf` | Working — splits MGF into ≤ N GiB chunks |
| `change_mgf_headers` | `mgf_ops.cli.change:change_mgf_headers` | Working — rewrites TITLE lines via regex→format; outputs to STDOUT |
| `cut_and_index_precursors` | `mgf_ops.cli.cut_and_index_precursors:main` | Working |
| `postprocess_pmsms` | `mgf_ops.cli.postprocess_pmsms:main` | Working |

---

## TOML config for `msms2mgf`

`parse_inputs_for_msms2mgf()` reads the `[msms2mgf]` section if present; otherwise
uses the whole file (`config.get("msms2mgf", config)`).

```toml
[msms2mgf.fragments]
mz_digits = 3                      # decimal places for m/z
mz_intensity_separator = " "      # bytes between m/z and intensity on each line
after_intensity = "\n"            # bytes after intensity (typically newline)
end_ions = "END IONS\n\n"         # appended after each spectrum

[msms2mgf]
ms1_header_sql = """
    SELECT header, header_len FROM precursors ...
"""
# DuckDB SQL; run against a registered `precursors` table.
# Must produce columns: `header` (str), `header_len` (int).
```

Bundled configs (`sage2peaks_nointensity.toml`, `sage2peaks_withintensity.toml`) are
for `change_mgf_headers` and contain `in_header_format`, `out_header_format`,
`fast_check` fields.

---

## Key performance patterns

**Hash-then-bytes (m/z)**
`count_floats` bins all float m/z values into integer buckets (`int(mz * 10^digits)`).
Non-zero buckets become the unique m/z set. DuckDB formats each as a fixed-precision
string. ASCII bytes are stored in a single contiguous array; `hash_to_ascii_idx` is a
cumulative-sum index into it. Hot loops look up `int_mz_to_hash[int(mz * mult)]` then
slice `mz_ascii[s:e]`.

**Hash-then-bytes (intensity)**
`count_per_batch` (parallel histogram via `divide_indices` + `prange`) counts unique
integer intensities across thread-local chunks then sums. Same indexing scheme as m/z.

**Exact allocation**
`count_ascii_per_fragment_pair` (`@njit parallel=True`) counts bytes per spectrum
before any allocation. `np.memmap` is opened at exactly `sum(byte_counts)`.

**Numba parallelism**
`count_ascii_per_fragment_pair` and `fill_mgf` (indexing.py) both use
`@njit(parallel=True)` + `numba.prange`. `write_spectra` in `writers.py` is the legacy
equivalent for the cluster-based format.

**Progress reporting**
`ProgressBar` from `numba_progress` is passed into `@njit` functions and called with
`progress.update(1)` inside `prange` loops.

**Config/data access**
`DotDict` / `DotDict.Recursive` from `dictodot` for dotted attribute access on nested
dicts loaded from TOML or `mmappet.open_dataset_dct()`.

---

## Key return shapes

`index_precursors(precursors, ms1_header_sql)` → `DotDict`:
- `.size` — header byte length per precursor
- `.idx` — cumulative-sum index (length N+1) into `.ascii`
- `.ascii` — all header bytes concatenated
- `.header` — header strings as numpy array

`get_mz_indexes(mzs, mz_digits)` → `DotDict`:
- `.int_mz` — unique rounded integer m/z values
- `.int_mz_to_hash` — maps `int(mz * mult)` → hash index (reuses `int_mz_counts` array)
- `.hash_to_ascii_idx` — cumulative-sum index into `.ascii`
- `.ascii`, `.str`, `.str_len` — formatted strings and their bytes

`get_intensity_indexes(intensities)` → same shape as above but for intensities.

---

## Development notes

- Inline tests (`test_get_mz_indexes`, `test_get_intensity_indexes`,
  `test_len_of_integer_part`) live in `indexing.py` and `math.py` — runnable with
  `python -m pytest` or directly.
- End-to-end test: `tests/test_msms2mgf.py` — run with `../../venvs/common/bin/python tests/test_msms2mgf.py`.
  Contains `run_test()` (single-charge) and `run_multicharge_test()` (multi-charge via `charges` column).
- `boundscheck=True` on Numba functions is intentional for development correctness;
  remove for production builds if performance matters.
- `__dev/` contains exploratory scripts and is not part of the installed package.

---

## Multi-charge support (`charges` column)

When precursors have a `charges` column instead of `charge`, one integer encodes multiple
possible charges as its decimal digits (e.g. `1246` → charges 1, 2, 4, 6).

**Expand-then-reuse** pattern: `expand_precursors_by_charges()` in `writers.py` calls
`explode_charge_codes()` (Numba, `charges.py`) to decode each digit and duplicate the
precursor row. The expanded DataFrame has a `charge` column and the same
`fragment_spectrum_start`/`fragment_event_cnt` values, so the downstream pipeline writes
identical fragment peaks for each charge automatically.

Use `msms2mgf_multicharge()` instead of `msms2mgf()` when the precursor table has a
`charges` column. CLI entry point: `msms_multiple_charge`.
