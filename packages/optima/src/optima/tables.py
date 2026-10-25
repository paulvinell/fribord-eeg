"""Parameter tables recovered from the vendor driver, plus pure conversions.

Data only, no device or transport dependencies. A dependent parameter is stored in the order of its
superior's domain entries, which is what pairs a context-dependent wire code with its meaning.
"""

from __future__ import annotations


# Impedance / input-continuity test, from parameters 0x90 and 0x91 plus the driver's display
# code at 0x10008250. The test streams through NOT_DATA exactly like a measurement, but:
#   - every enabled channel whose function is < 4 reports once per phase group (no
#     decimation — the driver leaves all interleave masks at 0 in test mode)
#   - the rate is fixed at rate exponent 2 = 3.90625 Hz
#   - values scale with the channel's kOhm range, not its measurement range
TEST_RATE_EXPONENT = 2


TEST_RATE_HZ = 3.90625


TEST_UNIT = "kOhm"


# Parameter 0x91 is read-only and derived from the channel's range setting; the values its
# variants carry are exactly the kOhm measurement ranges.
RANGE_TEST_KOHM = {31.25, 125.0, 1000.0}


TEST_RANGE_DEFAULT = 125.0


def test_scale(range_kohm: float = TEST_RANGE_DEFAULT) -> float:
    """kOhm per count during the impedance test.

    The range is read-only and derived from the channel's configuration; 125 kOhm is the
    common default, with 31.25 and 1000 appearing on other variants. Confirm against the
    driver's own Test tab reading before trusting an absolute number.
    """
    return range_kohm / ADC_FULL_SCALE


# Channel function encoding (parameter 0x8C), from its domain table. The sample decoder's
# `< 4` and `!= 7` tests are tests against this enum.
FUNC_VOLTAGE = 0


FUNC_RESISTANCE = 1


FUNC_CONDUCTANCE = 2


FUNC_TEMPERATURE = 3


FUNC_NIR_HEG = 4


FUNC_PIR_HEG = 5


FUNC_CURRENT = 6


FUNC_BUTTON = 7


FUNC_NAME = {
    FUNC_VOLTAGE: "voltage", FUNC_RESISTANCE: "resistance", FUNC_CONDUCTANCE: "conductance",
    FUNC_TEMPERATURE: "temperature", FUNC_NIR_HEG: "nIR HEG", FUNC_PIR_HEG: "pIR HEG",
    FUNC_CURRENT: "current", FUNC_BUTTON: "button",
}


# Total sample throughput limit, from the vendor manual's technical data: the sum of sample
# rates across all active channels must not exceed this. The device signals a violation by
# flashing its measurement indicator red ("too wide data stream").
MAX_SAMPLE_STREAM = 5000


# Channel profile (parameter 0x86) — host-side only, never sent to the device, but it is the
# vocabulary the vendor's UI uses and a useful shorthand for a sane per-signal setup.
PROFILE = {
    0: "User defined", 1: "EEG", 2: "EMG", 3: "ECG for HRV", 4: "GSR", 5: "TEMP",
    6: "EEG 250sps", 7: "SCP", 8: "EOG", 9: "RESP belt", 10: "nIR HEG", 11: "pIR HEG",
    12: "BVP", 13: "Button", 14: "ECG",
}


# Frequency characteristic (parameter 0x8D). Bandwidth and group delay from the vendor docs.
FREQ_CHAR = {
    0: "linear phase sharp (BW 0.4*sr, 19-sample delay)",
    1: "linear phase mild (BW 0.3*sr, 11-sample delay)",
}


# Common reference (parameter 0x92). Optima+ calls 1/2 internal/external; older Optima models
# label the same codes "Reference 1" and "Reference 2".
COMMON_REF = {0: "no connection", 1: "internal connection", 2: "external connection"}


# EEG cap connection (parameter 0x93): 0 = none, 8..15 select a 10-20 site. Which site each
# code means is *variant-dependent* — the same code maps to different sites on different
# channels and models (code 8 appears as C3, C4 and Cz across variants), so there is no single
# table. Read it back from the device rather than assuming.
EEG_CAP_NONE = 0


EEG_CAP_CODES = range(8, 16)


# Full-scale divisor for a sample, from the driver's own conversion at 0x10024450 (32768.0),
# corroborated by the adjacent 100/32768 constant used for its percentage bar. Consistent
# with ADC resolution = 16 bits and "Range minimum" = -range (symmetric full scale).
ADC_FULL_SCALE = 32768


# Measurement range (parameter 0x87) wire value -> full-scale magnitude, collected from the
# domain tables of every model. The unit depends on the channel function.
#
# CAVEAT: the same wire value can mean different magnitudes on different models and under
# different channel functions — the driver picks a domain variant based on the channel's
# other settings. These are the values observed across all Optima models; for a channel
# whose function is not plain voltage, confirm against the device before trusting them.
# Everything below is keyed by channel function, because the driver's parameter dependency
# graph makes function the superior parameter for range, rate, filtering and reference: the
# variants of each dependent are stored in the same order as the function's domain entries, so
# the pairing is exact rather than inferred. Read off channel 0's descriptor array of the
# Optima+ 4 (model 9); the other channels carry identical tables apart from EEG cap sites.
#
# (range wire code -> full-scale magnitude), unit
RANGE_BY_FUNCTION: dict[int, tuple[dict[int, float], str]] = {
    FUNC_VOLTAGE: ({0: 800.0, 3: 6000.0, 5: 24000.0}, "uV"),
    FUNC_RESISTANCE: ({0: 31.25, 2: 125.0, 5: 1000.0}, "kOhm"),
    FUNC_CONDUCTANCE: ({5: 20.0, 2: 160.0, 0: 640.0}, "uS"),
    FUNC_TEMPERATURE: ({5: 30.0, 2: 120.0}, "°C"),
    FUNC_NIR_HEG: ({5: 200.0}, "%"),
    FUNC_PIR_HEG: ({3: 50.0}, "°C"),
    FUNC_CURRENT: ({1: 400.0, 2: 2000.0, 3: 25000.0}, "nA"),
}


# "Range minimum" (0x88) paired exactly with its range, via the dependency ordering.
# This is the *specified valid lower bound* of the measurement, NOT a code offset — see
# valid_span() and the note on scaling below.
RANGE_MIN_BY_FUNCTION: dict[int, dict[int, float]] = {
    FUNC_VOLTAGE: {0: -800.0, 3: -6000.0, 5: -24000.0},      # symmetric
    FUNC_RESISTANCE: {0: 0.0, 2: 0.0, 5: 0.0},               # zero-based
    FUNC_CONDUCTANCE: {5: 1.0, 2: 8.0, 0: 32.0},             # range/20
    # NOTE: this temperature pairing follows the descriptor ordering, but the vendor manual
    # quotes the span as "-18..120 °C", which would pair -18 with the 120 range instead — the
    # reverse. Treat temperature spans as unconfirmed; the scale factor is unaffected.
    FUNC_TEMPERATURE: {5: -18.0, 2: 21.0},
    FUNC_NIR_HEG: {5: 0.0},
    FUNC_PIR_HEG: {3: 0.0},
    FUNC_CURRENT: {1: -400.0, 2: -2000.0, 3: -25000.0},      # symmetric
}


# Sample rates each function permits (rate exponents). Only voltage is free to choose.
RATES_BY_FUNCTION: dict[int, set[int]] = {
    FUNC_VOLTAGE: {6, 7, 8, 9, 10, 11},   # 62.5 .. 2000 sps
    FUNC_RESISTANCE: {4},                 # 15.625 sps
    FUNC_CONDUCTANCE: {4},
    FUNC_TEMPERATURE: {4},
    FUNC_NIR_HEG: {6},                    # 62.5 sps
    FUNC_PIR_HEG: {6},
    FUNC_CURRENT: {6},
}


DEFAULT_RATE_EXPONENT = 7  # 125 sps, the driver's default for a voltage channel


# Lower limit frequency (0x8A): (function, range code) -> {wire code: Hz}. Depends on both
# function and range, which is why the same code means 0.5 Hz on a voltage channel, 0.0 on a
# resistance channel and 0.35 on a current channel — context, not contradiction.
LOWER_LIMIT_BY_FUNCTION: dict[int, dict[int, dict[int, float]]] = {
    FUNC_VOLTAGE: {0: {0: 0.5, 2: 0.01},
                   3: {0: 0.5, 2: 0.01, 3: 0.0},
                   5: {0: 0.5, 2: 0.01, 3: 0.0}},
    FUNC_RESISTANCE: {0: {0: 0.0}, 2: {0: 0.0}, 5: {0: 0.0}},
    FUNC_CONDUCTANCE: {5: {0: 0.0}, 2: {0: 0.0}, 0: {0: 0.0}},
    FUNC_TEMPERATURE: {5: {0: 0.0}, 2: {0: 0.0}},
    FUNC_NIR_HEG: {5: {0: 0.0}},
    FUNC_PIR_HEG: {3: {3: 0.0}},
    FUNC_CURRENT: {1: {0: 0.35}, 2: {0: 0.35}, 3: {3: 0.0}},
}


# Frequency characteristic options, and the resulting upper corner as a fraction of the sample
# rate (parameter 0x8B, read-only). Confirms the vendor doc: sharp = 0.4*sr, mild = 0.3*sr.
FREQ_CHAR_BY_FUNCTION: dict[int, dict[int, float]] = {
    FUNC_VOLTAGE: {0: 0.4, 1: 0.3},
    FUNC_RESISTANCE: {0: 0.37},
    FUNC_CONDUCTANCE: {0: 0.37},
    FUNC_TEMPERATURE: {0: 0.038},
    FUNC_NIR_HEG: {1: 0.3},
    FUNC_PIR_HEG: {1: 0.3},
    FUNC_CURRENT: {0: 0.4, 1: 0.3},
}


# Impedance-test range (0x91) per (function, range code). A voltage channel always tests on
# 125 kOhm, which is what makes TEST_RANGE_DEFAULT right for EEG.
TEST_RANGE_BY_FUNCTION: dict[int, dict[int, float]] = {
    FUNC_VOLTAGE: {0: 125.0, 3: 125.0, 5: 125.0},
    FUNC_RESISTANCE: {0: 31.25, 2: 125.0, 5: 1000.0},
    FUNC_CONDUCTANCE: {5: 1000.0, 2: 125.0, 0: 31.25},
    FUNC_TEMPERATURE: {5: 1000.0, 2: 125.0},
    FUNC_NIR_HEG: {5: 1000.0},
    FUNC_PIR_HEG: {3: 1000.0},
    FUNC_CURRENT: {1: 1000.0, 2: 1000.0, 3: 1000.0},
}


# Common reference is only selectable on a voltage channel; every other function forces 0.
COMMON_REF_BY_FUNCTION: dict[int, set[int]] = {
    FUNC_VOLTAGE: {0, 1, 2},
    **{f: {0} for f in (FUNC_RESISTANCE, FUNC_CONDUCTANCE, FUNC_TEMPERATURE,
                        FUNC_NIR_HEG, FUNC_PIR_HEG, FUNC_CURRENT)},
}


# Driver defaults per function: mains notch on only for voltage; sum-disconnected set for the
# functions whose sensors have no galvanic skin contact.
NOTCH_DEFAULT = {FUNC_VOLTAGE: 1}


SUM_DISCONNECTED_DEFAULT = {FUNC_TEMPERATURE: 1, FUNC_NIR_HEG: 1, FUNC_PIR_HEG: 1,
                            FUNC_CURRENT: 1}


# EEG cap sites (parameter 0x93), per channel — each channel exposes a different subset of the
# 10-20 montage, which is why there is no single table. Only the voltage function offers sites;
# every other function forces 0 (None). Optima+ 4.
EEG_CAP_SITES: dict[int, dict[int, str]] = {
    0: {0: "None", 8: "Cz", 9: "F3", 10: "F4", 11: "F7", 12: "Fp1", 13: "Fpz", 14: "Fz", 15: "T3"},
    1: {0: "None", 8: "C3", 9: "Cz", 10: "F4", 11: "F7", 12: "F8", 13: "Fp2", 14: "Pz", 15: "T4"},
    2: {0: "None", 8: "C3", 9: "C4", 10: "Cz", 11: "O1", 12: "P3", 13: "P4", 14: "T4", 15: "T5"},
    3: {0: "None", 8: "C4", 9: "O1", 10: "O2", 11: "Oz", 12: "P3", 13: "P4", 14: "T5", 15: "T6"},
}


def cap_code(channel: int, site: str) -> int:
    """EEG cap wire code for a 10-20 site on a given channel.

    Sites are not interchangeable between channels: Cz is code 8 on channel 0 but 9 on
    channel 1 and 10 on channel 2, and no channel offers every site.
    """
    sites = EEG_CAP_SITES.get(channel)
    if sites is None:
        raise ValueError(f"no cap table for channel {channel}")
    for code, name in sites.items():
        if name.lower() == site.lower():
            return code
    available = ", ".join(n for n in sites.values() if n != "None")
    raise ValueError(f"channel {channel} cannot reach {site}; it offers {available}")


def rate_hz(exponent: int) -> float:
    """Sample rate in Hz for a stored rate exponent.

    From the driver's domain table: 6 -> 62.5, 7 -> 125, 8 -> 250, 9 -> 500, 10 -> 1000,
    11 -> 2000. So rate = 62.5 * 2**(exponent - 6), which is why the interleave arithmetic
    can treat rate differences as bit shifts.
    """
    return 62.5 * 2 ** (exponent - 6)


def rate_exponent(hz: float) -> int:
    """Inverse of rate_hz(), for building a config."""
    from math import log2, isclose
    e = 6 + log2(hz / 62.5)
    if not isclose(e, round(e), abs_tol=1e-9):
        raise ValueError(f"{hz} Hz is not a supported rate (62.5 * 2^n)")
    return round(e)
