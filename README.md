### mgf_ops

A Python module with tools to deal with MGF operations.
Why on Earth are they not binary?

The current version of the tool works with 2 formats.

1. The precursor-clusters-fragment_clusters-edges format, where precursors and fragments are kept in two parquet files and edges are kept in .mmappet format.

```bash
matteo@fuckafd$ write_mgf --help
Usage: write_mgf [OPTIONS] PRECURSOR_CLUSTER_STATS FRAGMENT_CLUSTER_STATS
                 MATCHES CONFIG OUT_MGF

Options:
  --threads_cnt INTEGER  [default: 16]
  --verbose
  --help                 Show this message and exit.
```

2. The precursors-peaks fragment-spectra format, where precursors are again kept as a parquet file with two indexing columns into the .mmappet formated fragment-spectra folder that contains all fragment (m/z, intensity) pairs.

```bash
 matteo@fuckafd$ msms2mgf --help
Usage: msms2mgf [OPTIONS] MSMS_FOLDER CONFIG MGF_PATH

Options:
  --threads_cnt INTEGER  [default: 16]
  --dataset_name TEXT    [default: NA]
  --verbose
  --help                 Show this message and exit. 
``` 
