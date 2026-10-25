"""The configuration blob: field map, accessors, builders and validation."""

from __future__ import annotations

from .tables import ADC_FULL_SCALE, COMMON_REF_BY_FUNCTION, EEG_CAP_SITES, FREQ_CHAR_BY_FUNCTION, FUNC_BUTTON, FUNC_NAME, FUNC_VOLTAGE, LOWER_LIMIT_BY_FUNCTION, MAX_SAMPLE_STREAM, NOTCH_DEFAULT, RANGE_BY_FUNCTION, RANGE_MIN_BY_FUNCTION, RATES_BY_FUNCTION, SUM_DISCONNECTED_DEFAULT, rate_exponent, rate_hz


# Config blob geometry, from the per-model table at 0x100227a4. Identical across every
# Optima and Optima+ model: 16-byte header, 6 bytes per channel, 6 channels maximum.
CFG_BASE = 52


CFG_HEADER = 16


CFG_CHANNEL_STRIDE = 6


CFG_MAX_CHANNELS = 6


# An Optima+ 4 has FIVE channel records: four analog plus the button channel, so its config is
# 46 bytes, not 40. Confirmed on hardware — a 40-byte blob is rejected with status 2.
CFG_RECORDS = 5


# Link type encoding, from the driver's parameter-domain table.
LINK_BLUETOOTH = 1


LINK_USB = 2


# Complete config field map, from the driver's own parameter-mapping tables (general at
# geometry+0x0c, per-channel at geometry+0x10). Each entry is (byte offset, mask, shift);
# a value is read as (blob[offset] & mask) >> shift.
#
# Parameters the driver keeps host-side and never sends to the device (channel label,
# sensor info, channel profile, device id, link type, channel count) are absent here by
# design — they have mask 0 in the driver's tables.
HEADER_FIELDS = {
    "mains_frequency": (0, 0x01, 0),   # 0 = 50 Hz, 1 = 60 Hz
    "data_enable": (0, 0x02, 0),       # see below — REQUIRED
    "silent_mode": (0, 0x20, 0),
}


# `data_enable` is not in the driver's parameter tables at all; it was found by sweeping the
# header against real hardware. Behaviour, measured:
#   bits 1-2 both clear -> WRITE_CFG rejected outright with status 2 (invalid parameter)
#   bit 1 set (0x02)    -> accepted and streams normally
#   bit 2 only (0x04)   -> accepted, but the device emits NO NOT_DATA at all
#   both set (0x06)     -> accepted, streams identically to 0x02
# So bit 1 gates sample delivery and bit 2 is something the Optima does not implement (the
# driver's Lite family has extra feedback/score notification classes, which is a plausible
# but unconfirmed explanation). Always set bit 1.
HEADER_DATA_ENABLE = 0x02


CHANNEL_FIELDS = {
    "frequency_characteristic": (0, 0x07, 0),
    "common_reference": (0, 0x18, 3),  # 0 = none, 1 = internal, 2 = external
    "sum_disconnected": (0, 0x20, 0),
    "power_interference_filter": (0, 0x40, 0),
    "enable": (0, 0x80, 0),
    "channel_function": (1, 0xFF, 0),  # 0 = voltage; also resistance/conductance/temp/current
    "measurement_range": (2, 0xFF, 0),
    "sample_rate": (3, 0xFF, 0),       # exponent, see rate_hz()
    "lower_limit_frequency": (4, 0xFF, 0),
    "eeg_cap_connection": (5, 0xFF, 0),
}


CH_RATE_EXPONENT = CHANNEL_FIELDS["sample_rate"][0]


# "Range minimum" (parameter 0x88) is a read-only dependent of the measurement range, and it
# is NOT always -range. Observed minima per unit, as sets — the descriptor arrays do not store
# range and minimum adjacently, so which minimum pairs with which range is resolved at runtime
# by the driver's dependency machinery and could not be recovered statically:
#
#   uV    -150, -800, -1000, -6000, -8000, -24000   exactly negate the ranges -> symmetric
#   nA    -400, -2000, -25000                       exactly negate the ranges -> symmetric
#   kOhm  0                                         zero-based
#   uS    1, 8, 32                                  = range/20; neither symmetric nor zero-based
#   °C    -18, 0, 21                                offset spans, pairing unknown
#   %     0                                         zero-based
#
# So `raw * range / 32768` is only correct where the span is symmetric. It is what the driver
# itself computes for the impedance test (zero-based, where the device presumably emits
# non-negative codes), and it is exactly right for voltage and current. For conductance,
# temperature and HEG the true mapping is affine with a non-zero offset, and no code path in
# the driver reveals its form — so scaling those is refused rather than guessed.
def channel_range(cfg: bytes, channel: int) -> tuple[float, str]:
    """Return (full-scale magnitude, unit) for a channel."""
    func = get_field(cfg, CHANNEL_FIELDS["channel_function"], channel)
    wire = get_field(cfg, CHANNEL_FIELDS["measurement_range"], channel)
    if func not in RANGE_BY_FUNCTION:
        raise ValueError(f"channel function {func} ({FUNC_NAME.get(func, '?')}) has no range")
    table, unit = RANGE_BY_FUNCTION[func]
    if wire not in table:
        legal = ", ".join(f"{k}={v:g}" for k, v in table.items())
        raise ValueError(
            f"range code {wire} is not legal for {FUNC_NAME.get(func, func)}; "
            f"choose from {legal}"
        )
    return table[wire], unit


def channel_scale(cfg: bytes, channel: int) -> tuple[float, str]:
    """Return (units per count, unit name) for one channel.

    Multiply a raw int16 sample by the factor to get a physical value:

        value = raw * range / 32768

    This is uniform across channel functions. "Range minimum" is *not* an offset in this
    mapping — it is the specified valid lower bound of the measurement. The evidence is the
    impedance case: its minimum is 0, yet the driver's own conversion applies no offset at
    all, so the minimum cannot participate in the arithmetic. Consistent with that reading,
    voltage and current have minimum = -range (the whole symmetric span is usable), while
    conductance (range/20), temperature (-18 or 21) and HEG (0) simply specify a narrower
    valid band inside the same code range. Use valid_span() to clamp or flag readings.

    The uniform formula is verified only for the impedance path, where the driver computes it;
    for the rest it is a well-supported inference from the range/minimum structure.
    """
    rng, unit = channel_range(cfg, channel)
    return rng / ADC_FULL_SCALE, unit


def valid_span(cfg: bytes, channel: int) -> tuple[float, float, str]:
    """Return (specified minimum, full-scale maximum, unit) for a channel.

    Readings outside this band are out of specification even though they are representable —
    e.g. a conductance channel on the 160 µS range only claims validity from 8 µS up, so a
    lower reading usually means the sensor is not attached.
    """
    func = get_field(cfg, CHANNEL_FIELDS["channel_function"], channel)
    wire = get_field(cfg, CHANNEL_FIELDS["measurement_range"], channel)
    rng, unit = channel_range(cfg, channel)
    lo = RANGE_MIN_BY_FUNCTION.get(func, {}).get(wire)
    if lo is None:
        raise ValueError(f"no range minimum for function {func} range {wire}")
    return lo, rng, unit


def lower_limit_hz(cfg: bytes, channel: int) -> float:
    """Return the high-pass corner in Hz for a channel: 0.5, 0.01, or 0.0 for DC.

    The counterpart to `build_eeg_config(lower_limit_hz=...)`, and it exists because **a recording
    carries its corner and nothing could read it back out.** The corner decides whether a reward band
    measures the signal or the filter's roll-off, so a screen that assumes 0.5 Hz while replaying a
    capture made at 0.01 will warn about wiring that is in fact correct. The blob in the session header
    is the authority — it is what the replay decodes with — so ask it rather than the caller.

    Like the other accessors here it is per-channel, because the field is. `build_eeg_config` sets every
    channel alike and there is no control that varies them, but reading the channel rather than assuming
    uniformity costs nothing.
    """
    func = get_field(cfg, CHANNEL_FIELDS["channel_function"], channel)
    rng = get_field(cfg, CHANNEL_FIELDS["measurement_range"], channel)
    wire = get_field(cfg, CHANNEL_FIELDS["lower_limit_frequency"], channel)
    table = LOWER_LIMIT_BY_FUNCTION.get(func, {}).get(rng, {})
    if wire not in table:
        legal = ", ".join(f"{k}={v:g}Hz" for k, v in table.items()) or "none"
        raise ValueError(
            f"lower-limit code {wire} is not legal for {FUNC_NAME.get(func, func)} on range code "
            f"{rng}; choose from {legal}"
        )
    return table[wire]


def to_units(samples: list[int], cfg: bytes, channel: int) -> list[float]:
    """Convert raw samples for one channel into physical units."""
    factor, _ = channel_scale(cfg, channel)
    return [s * factor for s in samples]


def _single_bit(mask: int) -> bool:
    """The driver's own test for a boolean field: (mask >> 1) & mask == 0."""
    return (mask >> 1) & mask == 0


def _resolve(field: tuple[int, int, int], channel: int | None) -> tuple[int, int, int]:
    offset, mask, shift = field
    if channel is not None:
        offset += CFG_HEADER + channel * CFG_CHANNEL_STRIDE
    return offset, mask, shift


def get_field(blob: bytes, field: tuple[int, int, int], channel: int | None = None) -> int:
    """Read one config field. Pass `channel` for a per-channel field.

    Follows NdGetParam: a single-bit mask reads as a 0/1 boolean regardless of which bit it
    occupies; wider fields are masked and then right-shifted.
    """
    offset, mask, shift = _resolve(field, channel)
    raw = blob[offset] & mask
    if _single_bit(mask):
        return 1 if raw else 0
    return raw >> shift


def set_field(blob: bytes, field: tuple[int, int, int], value: int,
              channel: int | None = None) -> bytes:
    """Return a copy of `blob` with one config field replaced.

    Mirrors get_field's convention: booleans are passed as 0/1 and land on whichever bit the
    mask selects, rather than being shifted into place by the caller.
    """
    offset, mask, shift = _resolve(field, channel)
    if _single_bit(mask):
        raw = mask if value else 0
    else:
        raw = (value << shift) & 0xFF
        if raw & ~mask & 0xFF or (raw >> shift) != value:
            raise ValueError(f"value {value} does not fit mask 0x{mask:02x} at shift {shift}")
    out = bytearray(blob)
    out[offset] = (out[offset] & ~mask & 0xFF) | raw
    return bytes(out)


def validate_config(blob: bytes) -> list[str]:
    """Check a config against the driver's own legality tables. Returns a list of problems.

    The device rejects invalid configs with REP_WRITE_CFG status 2, and the driver has two
    specific complaints of its own: "At least one analog channel should be on" and
    "Unsupported device configuration". Catching these locally is cheaper than a round trip.
    """
    problems = []
    nchan = (len(blob) - CFG_HEADER) // CFG_CHANNEL_STRIDE
    if len(blob) != config_size(nchan):
        problems.append(f"length {len(blob)} is not header+{nchan}*{CFG_CHANNEL_STRIDE}")
    analog = 0
    for ch in range(nchan):
        if not get_field(blob, CHANNEL_FIELDS["enable"], ch):
            continue
        func = get_field(blob, CHANNEL_FIELDS["channel_function"], ch)
        rate = get_field(blob, CHANNEL_FIELDS["sample_rate"], ch)
        rng = get_field(blob, CHANNEL_FIELDS["measurement_range"], ch)
        low = get_field(blob, CHANNEL_FIELDS["lower_limit_frequency"], ch)
        fchar = get_field(blob, CHANNEL_FIELDS["frequency_characteristic"], ch)
        ref = get_field(blob, CHANNEL_FIELDS["common_reference"], ch)
        cap = get_field(blob, CHANNEL_FIELDS["eeg_cap_connection"], ch)
        if func != FUNC_BUTTON:
            analog += 1
        if func not in FUNC_NAME:
            problems.append(f"ch{ch}: unknown channel function {func}")
            continue
        if func == FUNC_BUTTON:
            continue
        if rate not in RATES_BY_FUNCTION.get(func, set()):
            allowed = ", ".join(f"{rate_hz(r):g}" for r in sorted(RATES_BY_FUNCTION[func]))
            problems.append(f"ch{ch}: {rate_hz(rate):g} Hz not allowed for "
                            f"{FUNC_NAME[func]} (allowed: {allowed})")
        if rng not in RANGE_BY_FUNCTION.get(func, ({}, ""))[0]:
            problems.append(f"ch{ch}: range code {rng} not legal for {FUNC_NAME[func]}")
        elif low not in LOWER_LIMIT_BY_FUNCTION.get(func, {}).get(rng, {}):
            allowed = LOWER_LIMIT_BY_FUNCTION.get(func, {}).get(rng, {})
            problems.append(f"ch{ch}: lower-limit code {low} not legal here "
                            f"(allowed: {allowed})")
        if fchar not in FREQ_CHAR_BY_FUNCTION.get(func, {}):
            problems.append(f"ch{ch}: frequency characteristic {fchar} not legal for "
                            f"{FUNC_NAME[func]}")
        if ref not in COMMON_REF_BY_FUNCTION.get(func, {0}):
            problems.append(f"ch{ch}: common reference {ref} not legal for "
                            f"{FUNC_NAME[func]} (only voltage channels can be referenced)")
        if cap and func != FUNC_VOLTAGE:
            problems.append(f"ch{ch}: EEG cap site set on a non-voltage channel")
        if cap and cap not in EEG_CAP_SITES.get(ch, {}):
            problems.append(f"ch{ch}: cap code {cap} is not a site this channel can reach")
    if analog == 0:
        problems.append("at least one analog channel must be enabled")
    stream = 0.0
    for ch in range(nchan):
        if get_field(blob, CHANNEL_FIELDS["enable"], ch):
            func = get_field(blob, CHANNEL_FIELDS["channel_function"], ch)
            if func != FUNC_BUTTON:
                stream += rate_hz(get_field(blob, CHANNEL_FIELDS["sample_rate"], ch))
    if stream > MAX_SAMPLE_STREAM:
        problems.append(f"total sample stream {stream:g} sps exceeds the device limit of "
                        f"{MAX_SAMPLE_STREAM} sps")
    if not get_field(blob, HEADER_FIELDS["data_enable"]):
        problems.append("header data_enable (byte 0 bit 1) must be set or the device rejects "
                        "the config with status 2")
    return problems


REFERENCE_MODES = {
    "linked": 1,      # "-" inputs joined inside the device: one reference electrode serves all
    "bipolar": 0,     # independent pairs; each channel needs its own two electrodes
    "external": 2,    # a second reference group, joined by a physical jumper
}


def build_eeg_config(channels: int = 4, nchan: int = CFG_RECORDS,
                     hz: float = 1000.0, range_uv: float = 800.0, mains_hz: int = 50,
                     reference: str = "linked", lower_limit_hz: float = 0.5,
                     silent: bool = True) -> bytes:
    """Build a ready-to-write config for EEG on the first `channels` analog channels.

    The device's EEG-cap-connection field is left at 0 and there is no way to set it here. It only
    means anything with Neurobit's own cap, whose wiring is what makes the codes per-channel and
    mutually exclusive; with loose electrodes you decide where they go and the device does not need
    to be told. `optima.EEG_CAP_SITES` documents the codes if you ever do use that cap.

    Defaults mirror the driver's own EEG choices: mains notch on, sum voltage connected,
    linear-phase-sharp characteristic, and the lowest lower-limit corner available (0.5 Hz).

    **`hz` defaults to 1000, not the 250 an EEG channel conventionally uses, and the reason is
    latency.** The device always packs exactly 8 sample groups into a frame regardless of rate, so
    a frame carries 8/rate seconds of signal and its oldest sample is already that old on arrival:
    32 ms at 250 sps against 8 ms at 1000. Nothing else about the pipeline changes for the worse —
    filter group delay in seconds is rate-independent, and the cFIR envelope estimate actually
    improves (r = 0.982 against 0.964 versus the non-causal reference, with effective delay 116 ms
    against 124). The cost is CPU: about 10% of one core against 4.6%, since there are four times
    as many frames to process.

    Four channels at 1000 sps is 4000 sps against the device's 5000 sps total limit, so this is the
    highest rate that leaves all four usable. 2000 sps is legal only with two channels.

    **`lower_limit_hz`** is the high-pass corner: 0.5 Hz, which is the vendor's own EEG choice, or 0.01 Hz
    for ultra-low-frequency work. Anything below 0.5 selects the 0.01 Hz code, since those are the only two
    corners a voltage channel offers short of DC (and DC needs a wider range). It matters because a reward
    band below the corner measures the filter's roll-off rather than the signal.

    **`reference` defaults to "linked"**, which joins every channel's "-" input inside the device so
    one reference electrode serves all of them: four signal electrodes plus one reference plus ground,
    no jumper. The vendor's own default is "bipolar", and the reason to differ is that cross-channel
    measures — coherence, phase — are only interpretable when the channels share a reference, and
    linked data can be re-referenced offline where bipolar data cannot. The cost is that one bad
    reference electrode contaminates all four channels at once.

    "bipolar" gives independent pairs and needs two electrodes per channel. "external" exists for
    running a second reference group joined by a physical jumper.

    **`silent` stops the device's own beeper, and defaults to on because the host owns every sound in a
    session.** The header bit was recovered with the rest of the General group and then never set, so the
    amplifier kept its factory behaviour and beeped on its own — audibly at every mode change, so stopping a
    session beeped at you. It is a setting and not the hardware.

    The reason to default it silent rather than expose it neutrally is that a beep, a
    click and an alert are three sounds with three jobs and merging any two loses information. A beeper
    inside the amplifier is a fourth sound that no setting on any screen describes, that `NFB_NO_OUTPUTS`
    cannot switch off, and that means something different from all three. Whatever it is signalling, the
    console already knows and can say properly.

    **The polarity is inferred, not measured.** The bit is `silent_mode` in the driver's own parameter
    tables, listed beside the indicator-support flags, so set = quiet is the reading the name supports — but
    the beeper is not observable from here, so this is the one claim in this function nothing has verified.
    Pass `silent=False` if it turns out backwards.
    """
    if mains_hz not in (50, 60):
        raise ValueError("mains_hz must be 50 or 60")
    if reference not in REFERENCE_MODES:
        raise ValueError(f"reference must be one of {sorted(REFERENCE_MODES)}, got {reference!r}")
    rng_table, _ = RANGE_BY_FUNCTION[FUNC_VOLTAGE]
    range_code = next((k for k, v in rng_table.items() if v == range_uv), None)
    if range_code is None:
        legal = ", ".join(f"{v:g}" for v in rng_table.values())
        raise ValueError(f"range {range_uv} uV not available; choose from {legal} uV")
    exponent = rate_exponent(hz)
    if exponent not in RATES_BY_FUNCTION[FUNC_VOLTAGE]:
        raise ValueError(f"{hz} Hz not available for a voltage channel")

    blob = bytes(config_size(nchan))
    blob = set_field(blob, HEADER_FIELDS["mains_frequency"], 1 if mains_hz == 60 else 0)
    blob = set_field(blob, HEADER_FIELDS["data_enable"], 1)
    blob = set_field(blob, HEADER_FIELDS["silent_mode"], 1 if silent else 0)
    # channel records beyond the analog ones (the button) are left disabled
    for ch in range(channels):
        if ch >= nchan:
            raise ValueError(f"channel {ch} beyond {nchan}-channel config")
        blob = set_field(blob, CHANNEL_FIELDS["enable"], 1, ch)
        blob = set_field(blob, CHANNEL_FIELDS["channel_function"], FUNC_VOLTAGE, ch)
        blob = set_field(blob, CHANNEL_FIELDS["measurement_range"], range_code, ch)
        blob = set_field(blob, CHANNEL_FIELDS["sample_rate"], exponent, ch)
        # 0.5 Hz (code 0) on every voltage range, or 0.01 Hz (code 2) when something below the usual
        # corner is actually being measured — the CFC protocol reaches "between ultra-low
        # frequencies and high frequencies such as gamma", and a 0.1 Hz reward against a 0.5 Hz corner
        # would be measuring the filter rather than the brain.
        blob = set_field(blob, CHANNEL_FIELDS["lower_limit_frequency"],
                         0 if lower_limit_hz >= 0.5 else 2, ch)
        blob = set_field(blob, CHANNEL_FIELDS["frequency_characteristic"], 0, ch)
        blob = set_field(blob, CHANNEL_FIELDS["power_interference_filter"],
                         NOTCH_DEFAULT.get(FUNC_VOLTAGE, 0), ch)
        blob = set_field(blob, CHANNEL_FIELDS["sum_disconnected"],
                         SUM_DISCONNECTED_DEFAULT.get(FUNC_VOLTAGE, 0), ch)
        blob = set_field(blob, CHANNEL_FIELDS["common_reference"],
                         REFERENCE_MODES[reference], ch)
    problems = validate_config(blob)
    if problems:
        raise ValueError("built an invalid config: " + "; ".join(problems))
    return blob


def contact_check_config(blob: bytes, nchan: int = CFG_RECORDS) -> bytes:
    """`blob` with every channel taken off the SUM entity, for the impedance test.

    **This corrects a reading, not a preference.** The SUM entity ties the connected channels together
    inside the device, and during the impedance test that node couples them: a channel with a floating
    input drags the reading of a channel that has an electrode. Measured on hardware, one electrode at C4
    against one reference, four channels enabled and bipolar wiring — ch0 read `+25.41 / −41.56` with every
    channel summed and `+5.52 / −10.41` with the empty ones taken off the sum, against `+5.52 / −10.56`
    with only ch0 enabled at all. So off-the-sum is exact and the summed reading was out by 4×.

    Under linked wiring it recovers most but not all of it (21.38 → 13.09 against 5.58 with one channel),
    because linked *also* joins every "−" input, and that path is the wiring rather than the sum. Disabling
    the empty channels is what closes the rest — see `nfb.Device.narrow_to_wired`.

    Applied only to the test, so the acquisition config stays byte-identical: SUM is a measurement entity
    this driver never reads samples from, but it is not this function's business to decide that for a
    measurement.
    """
    for ch in range(nchan):
        blob = set_field(blob, CHANNEL_FIELDS["sum_disconnected"], 1, ch)
    return blob


def describe_config(blob: bytes) -> str:
    """Human-readable dump of a config blob, for working out what the device came with."""
    lines = [f"config: {len(blob)} bytes, header {CFG_HEADER}, "
             f"{(len(blob) - CFG_HEADER) // CFG_CHANNEL_STRIDE} channel records"]
    for name, f in HEADER_FIELDS.items():
        lines.append(f"  {name} = {get_field(blob, f)}")
    nchan = (len(blob) - CFG_HEADER) // CFG_CHANNEL_STRIDE
    for ch in range(nchan):
        vals = {n: get_field(blob, f, ch) for n, f in CHANNEL_FIELDS.items()}
        if not vals["enable"]:
            lines.append(f"  ch{ch}: off")
            continue
        func = FUNC_NAME.get(vals["channel_function"], vals["channel_function"])
        try:
            factor, unit = channel_scale(blob, ch)
            fs = factor * ADC_FULL_SCALE
            scale = f"+/-{fs:g}{unit} ({factor:.6g} {unit}/count)"
        except ValueError as e:
            scale = f"range code {vals['measurement_range']} ({e})"
        extras = []
        try:
            lo, hi, unit = valid_span(blob, ch)
            if lo != -hi:
                extras.append(f"valid {lo:g}..{hi:g}{unit}")
        except ValueError:
            pass
        if extras:
            scale += " [" + ", ".join(extras) + "]"
        lines.append(f"  ch{ch}: {func} {rate_hz(vals['sample_rate']):g}Hz {scale} "
                     f"ref={vals['common_reference']} notch={vals['power_interference_filter']}")
    return "\n".join(lines)


def config_size(records: int = CFG_RECORDS) -> int:
    """CMD_WRITE_CFG payload size for a given number of channel *records*.

    Note this counts records, not usable analog channels: an Optima+ 4 has 5 records (4 analog
    + button) and a 46-byte config.
    """
    return CFG_BASE - CFG_CHANNEL_STRIDE * (CFG_MAX_CHANNELS - records)
