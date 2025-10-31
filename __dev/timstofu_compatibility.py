%load_ext autoreload
%autoreload 2

import pandas as pd
import resource


from mgf_ops.readers import parse_inputs_for_msms2mgf
from mgf_ops.writers import msms2mgf


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 4)


parsed_inputs = parse_inputs_for_msms2mgf(
    msms_folder="/home/matteo/tmp/F9477_top_prob_pseudo.msms",
    config="/home/matteo/Projects/timstofu/ionmaiden_pipeline/configs/default.toml",
    mgf_path="/home/matteo/tmp/F9477.mgf",
    threads_cnt=16,
    dataset_name="F9477",
    verbose=True
)
locals().update(**parsed_inputs)
msms2mgf(**parsed_inputs)


# spectrum_direct = get_direct_spectrum(
#     spec_id, config.fragments.mz_digits, pseudomsms, headers, config.endions
# )
# with open(f"/tmp/spec_{spec_id}.mgf", "w") as f:
#     f.write(spectrum_direct)

# pprint(spectrum_direct)
# len(spectrum_direct)

# spec_id = 10253

# buggy_spec = ascii2str(mgf[spectrum_idx[spec_id] : spectrum_idx[spec_id + 1]])
# with open(f"/tmp/buggy_spec_{spec_id}.mgf", "w") as f:
#     f.write(buggy_spec)


# would be comfy to have something like a final idx
# precursor_to_frag_idx = pseudomsms.precursors.fragment_spectrum_start.to_numpy()
# precursor_to_frag_cnt = pseudomsms.precursors.fragment_event_cnt.to_numpy()
# headers_idx = headers.idx
# headers_ascii = headers.ascii
# fragment_mz_digits = config.fragments.mz_digits
# mzs = pseudomsms.fragments.mz
# int_mz_to_hash = MZ.int_mz_to_hash
# mz_hash_to_ascii = MZ.hash_to_ascii_idx
# mz_ascii = MZ.ascii
# intensities = pseudomsms.fragments.intensity
# intensity_to_hash = INTENSITY.intensity_to_hash
# intensity_hash_to_ascii = INTENSITY.hash_to_ascii_idx
# intensity_ascii = INTENSITY.ascii
# prec_idx = 0
