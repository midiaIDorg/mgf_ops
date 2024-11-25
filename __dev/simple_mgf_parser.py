from collections import Counter

from tqdm import tqdm

mgf_path_new = "partial/G8027/G8045/MS1@tims@1fd37e91592@default@fast@default@MS2@tims@1fd37e91592@default@fast@default/matcher@prtree@narrow/rough@fast.mgf_fast"


def iter_spectra(
    codelines,
    _start_tag: str = "BEGIN IONS",
    _stop_tag: str = "END IONS",
) -> list[dict]:
    recording = False
    for line in codelines:
        line = line.strip()
        # Detect sections based on the keywords
        if line.startswith(_start_tag):
            recording = True
            buffer: list[str] = []
        elif line.startswith(_stop_tag):
            assert recording
            recording = False
            yield buffer
        elif recording:
            buffer.append(line)
        else:
            pass
    assert not recording


def parse_line_spectrum(line_spectrum):
    header = line_spectrum[:4]
    peaks = []
    for l in line_spectrum[4:]:
        mz, intensity = l.split(" ")
        peaks.append((float(mz), int(intensity)))
    return header, peaks


def parse_mgf(mgf_path, sort_peaks=False):
    MS1_ClusterIDs = []
    peak_cnts = []
    peak_list = []
    with open(mgf_path, "r") as file:
        for line_spectrum in tqdm(iter_spectra(file)):
            header, peaks = parse_line_spectrum(line_spectrum)
            MS1_ClusterID = int(header[0].split(".", 2)[1])
            MS1_ClusterIDs.append(MS1_ClusterID)
            if sort_peaks:
                peaks.sort()
            peak_cnt = len(peaks)
            peak_cnts.append(peak_cnt)
            peak_list.append(peaks)
    return np.array(MS1_ClusterIDs), np.array(peak_cnts), peak_list


mgf_path_new = "partial/G8027/G8045/MS1@tims@1fd37e91592@default@fast@default@MS2@tims@1fd37e91592@default@fast@default/matcher@prtree@narrow/rough@fast.mgf_fast"

mgf_path_old = "partial/G8027/G8045/MS1@tims@1fd37e91592@default@fast@default@MS2@tims@1fd37e91592@default@fast@default/matcher@prtree@narrow/rough@default.mgf"

MS1_ClusterIDs_new, peak_cnts_new, peak_list_new = parse_mgf(mgf_path_new)
MS1_ClusterIDs_old, peak_cnts_old, peak_list_old = parse_mgf(mgf_path_old)

np.all(MS1_ClusterIDs_new == MS1_ClusterIDs_old)
MS1_ClusterIDs = MS1_ClusterIDs_new

np.all(peak_cnts_new == peak_cnts_old)
np.sum(peak_cnts_new == peak_cnts_old)
len(peak_cnts_new)

sum(peak_cnts_new)
sum(peak_cnts_old)

diffs = MS1_ClusterIDs[peak_cnts_new != peak_cnts_old]
diff = diffs[0]

diff_idx = (MS1_ClusterIDs == diff).nonzero()[0][0]

assert diff == MS1_ClusterIDs[diff_idx]
new_frags = peak_list_new[diff_idx]
old_frags = peak_list_old[diff_idx]

new_frags.sort()

len(new_frags)
len(old_frags)

Counter(cnt for k, cnt in Counter(intensity for _, intensity in new_frags).items())
Counter(cnt for k, cnt in Counter(intensity for _, intensity in old_frags).items())

{intensity for _, intensity in new_frags} - {intensity for _, intensity in old_frags}
{intensity for _, intensity in old_frags} - {intensity for _, intensity in new_frags}


set(peak_cnts_old[peak_cnts_new != peak_cnts_old])

new_frags.sort(key=lambda x: x[1], reverse=True)
new = new_frags[:1024].copy()
new.sort()

intensities_new = np.array([inten for _, inten in new])
intensities_old = np.array([inten for _, inten in old_frags])

np.all(intensities_new == intensities_old)
