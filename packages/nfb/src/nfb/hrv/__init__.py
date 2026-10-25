"""Heart rate variability — a sibling of the EEG path, not a layer on it.

**Beats are not samples.** An RR interval arrives roughly once a second, irregularly spaced, and carries
its own timestamp; a `Block` is a run of evenly spaced samples with an index. `BeatSource` therefore does
not satisfy `stream.SampleSource` and should not be made to. Everything above the two of them —
thresholds, targets, the feedback signal, the session clock — is shared, and that is the right seam.

**Optional, not a prerequisite.** Training HRV before EEG is a clinical position, and not one the software
should enforce. So this is a protocol you can choose.

**The target is the shape of the spectrum, not a score.** The goal is a single spectral peak at 0.1 Hz,
reached by breathing in for five seconds and out for five while watching the heart rate: in as it
accelerates, out as it slows. So the instruments are a tachogram and a power spectrum over 0–0.4 Hz, laid
out like the EEG display.

**0.1 Hz is where that starts, not where it is nailed down.** The resonant frequency of the baroreflex is a
person's own — taller adults resonate lower, and published resonance protocols find people between 4.5 and 7
breaths a minute. The target and the width of the band around it are therefore settings, carried on each
`Spectrum` so that every number scored against them and the marked band on the screen move together. See
`beats.TARGET_HZ`.

One failure is the reason a single coherence number is not enough. The spectrum can **split**
into a high-frequency parasympathetic peak and a very-low-frequency sympathetic one with the 0.1 Hz peak
gone, and this can happen while the EEG appears to improve. Any measure of "power in a band" rises in both
cases. `Spectrum.split` is the detector for it.
"""

from .artifacts import ArtifactFilter, ArtifactParams, Verdict
from .beats import (
    COHERENCE_TOP_HZ, HF_BAND, NARROW_BAND, NARROW_HALF_HZ, NARROW_HALF_RANGE_HZ, RESONANCE_BAND,
    RR_SCALE_MS, RR_SCALE_SPEC, TARGET_HZ, TARGET_RANGE_HZ, VLF_BAND,
    Beat, BeatState, HRVProcessor, RrScaleResolver, Spectrum, narrow_band, parse_hr_measurement,
    parse_hr_raw,
)
from .metrics import DEFAULT_METRIC, HRV_METRICS, LF_BAND, MetricSpec, metric_value
from .sources import (
    BeatSource, BleBeats, SimulatedBeats, UsbPpgBeats, beat_source, find_usb_port,
)

__all__ = [
    "Beat", "BeatState", "HRVProcessor", "Spectrum", "parse_hr_measurement", "parse_hr_raw",
    "RESONANCE_BAND", "NARROW_BAND", "VLF_BAND", "HF_BAND", "COHERENCE_TOP_HZ",
    "TARGET_HZ", "TARGET_RANGE_HZ", "NARROW_HALF_HZ", "NARROW_HALF_RANGE_HZ", "narrow_band",
    "RrScaleResolver", "RR_SCALE_SPEC", "RR_SCALE_MS",
    "ArtifactFilter", "ArtifactParams", "Verdict",
    "HRV_METRICS", "DEFAULT_METRIC", "LF_BAND", "MetricSpec", "metric_value",
    "BeatSource", "BleBeats", "SimulatedBeats", "UsbPpgBeats", "beat_source", "find_usb_port",
]
