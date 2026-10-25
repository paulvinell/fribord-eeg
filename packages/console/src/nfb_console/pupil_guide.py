"""A positioning guide for the eye tracker: a box with the eyes shown in it, and the distance.

The tracker reports where each eye sits in its field of view, 0..1 on each axis, and how far away the
face is. This draws that as a box with a dot per eye and a centre target, so a person can line
themselves up — the same job the raw camera would do, from data the tracker gives without the camera
stream. The device must sit below the screen, aimed at the face, about an arm's length away.
"""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from . import theme as t

NEAR_MM, FAR_MM = 450.0, 950.0
IDEAL_MM = 650.0
CENTRE_ZONE = 0.18          # how far from the middle still counts as centred, in box fractions


def _centred(position) -> bool:
    eyes = [e for e in (position.left, position.right) if e is not None]
    if not eyes:
        return False
    mx = sum(e[0] for e in eyes) / len(eyes)
    my = sum(e[1] for e in eyes) / len(eyes)
    near = position.distance_mm is None or NEAR_MM <= position.distance_mm <= FAR_MM
    return abs(mx - 0.5) <= CENTRE_ZONE and abs(my - 0.5) <= CENTRE_ZONE and near


def guide_shapes(position, width: float, height: float) -> list:
    """The box, a centre target, and a dot per eye. The eyes are drawn mirrored, so moving your head one
    way moves the dots the same way, like a mirror."""
    pad = 6
    bw, bh = width - 2 * pad, height - 2 * pad
    good = position is not None and _centred(position)
    frame = ft.Paint(color=t.BORDER, style=ft.PaintingStyle.STROKE, stroke_width=1.5)
    shapes = [cv.Rect(pad, pad, bw, bh, border_radius=10, paint=frame),
              cv.Circle(pad + bw / 2, pad + bh / 2, min(bw, bh) * CENTRE_ZONE,
                        paint=ft.Paint(color=t.HAIRLINE, style=ft.PaintingStyle.STROKE,
                                       stroke_width=1))]
    colour = t.GOOD if good else t.ACCENT
    for eye in (position.left, position.right) if position is not None else ():
        if eye is None:
            continue
        ex = pad + (1.0 - min(1.0, max(0.0, eye[0]))) * bw      # mirror x
        ey = pad + min(1.0, max(0.0, eye[1])) * bh
        shapes.append(cv.Circle(ex, ey, 9, paint=ft.Paint(color=colour, style=ft.PaintingStyle.FILL)))
    return shapes


def guide_status(position) -> tuple[str, str]:
    """A one-line verdict and a tone name (muted/accent/good/bad)."""
    if position is None or not position.seen:
        return "the tracker does not see your eyes — sit in front of it, below the screen", "bad"
    d = position.distance_mm
    if d is not None and d < NEAR_MM:
        return f"a little far back — {d / 10:.0f} cm, move closer", "accent"
    if d is not None and d > FAR_MM:
        return f"a little close — {d / 10:.0f} cm, move back", "accent"
    # The trackbox position tells left/right/up; without it, a good distance is all there is to say.
    if position.left is None and position.right is None:
        return (f"eyes seen, {d / 10:.0f} cm — good" if d else "eyes seen"), "good"
    if not _centred(position):
        return "centre your eyes in the box", "accent"
    return (f"centred, {d / 10:.0f} cm — good" if d else "centred — good"), "good"
