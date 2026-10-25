"""One place for colour and type.

Off-white with a trace of warmth in it, rather than screen-white. The warmth is deliberate but small:
a session is watched for twenty minutes at a stretch, often in a dim room next to someone with their
eyes closed, and pure white is the wrong amount of light in that room. Any more than a trace and the
ground starts competing with the accent and the good/warn/bad signals for attention, and those have to
carry meaning.

Names are shared with the earlier dark palette, so changing the whole look is a change to this file
plus `page.theme_mode`.
"""

from __future__ import annotations

BG = "#f1f1ee"          # window ground
SURFACE = "#fdfcfa"     # cards, panels — lifted off the ground, not quite white
SURFACE_HI = "#e9e7e2"  # hover, selected, empty bar troughs
BORDER = "#dad8d3"
HAIRLINE = "#e4e2dd"   # the lightest edge: panels, cards and filled controls carry this one

ACCENT = "#2e6bc2"      # the single accent
ACCENT_DIM = "#a1bee3"

TEXT = "#1c1b19"
MUTED = "#5f5b56"
FAINT = "#8f8b85"
ON_ACCENT = "#fdfcfa"

GOOD = "#248647"
WARN = "#a66306"
BAD = "#ba392b"

H1 = 26
H2 = 15
BODY = 13
SMALL = 11
VALUE = 24              # a card's live number: the one thing read from the chair
CODE = 40               # the big protocol letters


def card_colour(hovered: bool, enabled: bool) -> str:
    if not enabled:
        return SURFACE
    return SURFACE_HI if hovered else SURFACE


# Frequency → colour: `center_color(centre_hz)` looks a band's centre frequency
# up in one table shared by every view, so a band is the same colour on the spectrum, on its target card
# and in a legend. That consistency is the whole value — it means "the green one" is a thing you can say.
# The stops follow the convention the
# literature already uses: slow activity cool, alpha green, beta warm, gamma red.
FREQUENCY_STOPS = (
    (1.0, "#6247aa"),      # delta — violet
    (4.0, "#3a6fd0"),      # theta — blue
    (8.0, "#159a8a"),      # alpha — teal
    (12.0, "#2f9e44"),     # SMR — green
    (15.0, "#8a9b1f"),     # low beta — olive
    (20.0, "#c98a12"),     # beta — amber
    (30.0, "#cd6413"),     # high beta — orange
    (45.0, "#b8372c"),     # low gamma — red
    (60.0, "#8f2b3f"),     # gamma — deep red
)


def frequency_colour(hz: float) -> str:
    """Interpolate the stops above. Linear in frequency, so equal spacing on the spectrum axis is equal
    spacing in colour and a bin's hue can be read back as a frequency."""
    stops = FREQUENCY_STOPS
    if hz <= stops[0][0]:
        return stops[0][1]
    if hz >= stops[-1][0]:
        return stops[-1][1]
    for (lo_hz, lo_hex), (hi_hz, hi_hex) in zip(stops, stops[1:]):
        if lo_hz <= hz <= hi_hz:
            frac = (hz - lo_hz) / (hi_hz - lo_hz)
            lo = tuple(int(lo_hex[i:i + 2], 16) for i in (1, 3, 5))
            hi = tuple(int(hi_hex[i:i + 2], 16) for i in (1, 3, 5))
            mix = tuple(round(a + (b - a) * frac) for a, b in zip(lo, hi))
            return "#%02x%02x%02x" % mix
    return stops[-1][1]


def band_colour(low: float, high: float) -> str:
    """A band takes the colour of its centre, so one band is one colour in every view."""
    return frequency_colour((low + high) / 2)


# Amplitude → colour, for the cascading spectral display. A *different* job from the frequency ramp
# above and it must not be confused with it: on the spectrogram, frequency is the horizontal axis and
# colour carries magnitude, so a rainbow keyed to frequency would say the same thing twice and leave
# amplitude unrepresented.
#
# Starts at the panel's own surface rather than at black, because the ground here is off-white: "nothing
# at this frequency" should look like the panel, not like ink. Then cool, then hot — so the clinical read
# the manual describes, energy in the alpha skirts falling from 25 µV to under 5, is a colour change
# rather than a subtle brightness one.
AMPLITUDE_STOPS = (
    (0.00, "#fdfcfa"),
    (0.12, "#dbe6f2"),
    (0.30, "#7aa8d8"),
    (0.50, "#3f8f6b"),
    (0.68, "#c9a227"),
    (0.85, "#cd6413"),
    (1.00, "#8f2b3f"),
)
_AMP_POS = tuple(p for p, _ in AMPLITUDE_STOPS)
_AMP_RGB = tuple(tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for _, h in AMPLITUDE_STOPS)


def amplitude_colour(fraction: float) -> str:
    r, g, b = _amplitude_rgb(float(fraction))
    return "#%02x%02x%02x" % (round(r), round(g), round(b))


def _amplitude_rgb(fraction: float) -> tuple[float, float, float]:
    fraction = min(1.0, max(0.0, fraction))
    for i in range(len(_AMP_POS) - 1):
        lo, hi = _AMP_POS[i], _AMP_POS[i + 1]
        if lo <= fraction <= hi:
            f = (fraction - lo) / (hi - lo) if hi > lo else 0.0
            a, b = _AMP_RGB[i], _AMP_RGB[i + 1]
            return tuple(x + (y - x) * f for x, y in zip(a, b))
    return _AMP_RGB[-1]


def amplitude_ramp(fractions):
    """Vectorised form, for a whole spectrogram row at once. Returns an (n, 3) uint8 array.

    Vectorised here and nowhere else in the palette because this one is called on thousands of cells per
    frame, where the per-call overhead argument runs the other way.
    """
    import numpy as np

    values = np.clip(np.asarray(fractions, dtype=float), 0.0, 1.0)
    stops = np.asarray(_AMP_POS)
    table = np.asarray(_AMP_RGB, dtype=float)
    out = np.empty((values.size, 3), dtype=np.uint8)
    for channel in range(3):
        out[:, channel] = np.round(np.interp(values, stops, table[:, channel]))
    return out
