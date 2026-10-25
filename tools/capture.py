"""Take the picture. Starts `shots.py`, walks its routes, writes a PNG for each.

Usage: `make shots`, or `uv run python tools/capture.py --screen PAC --width 900`.

Deliberately not a golden-image diff. A pixel comparison fails on every innocent font or antialiasing
change and tells you *that* something moved rather than what is wrong; the rules in
`tests/test_layout` are the regression net. These are for looking at.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from urllib.parse import quote
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shots import SHOTS, route_of                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("SHOTS_PORT", "8553"))


def wanted(args) -> list:
    out = SHOTS
    if args.screen:
        want = {s.upper() for s in args.screen.split(",")}
        out = [s for s in out if s[0].upper() in want]
    if args.state:
        out = [s for s in out if s[0] == "launcher" or s[1] == args.state]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", help="only these protocol codes, comma separated")
    ap.add_argument("--state", choices=("cold", "fed"))
    ap.add_argument("--width", type=int, default=1240)
    ap.add_argument("--height", type=int, default=1400)
    ap.add_argument("--tab", type=int, help="a tab index instead of Train: 1 Spectrum, 2 Trends, 3 Feedback")
    ap.add_argument("--help-title", dest="help_title",
                    help="open the help dialog with this title before the shot")
    ap.add_argument("--out", default=str(ROOT / "shots"))
    args = ap.parse_args()

    shots = wanted(args)
    if not shots:
        sys.exit("nothing matches that selection")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is not installed — `uv sync --group shots` "
                 "then `uv run playwright install chromium`")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    server = subprocess.Popen([sys.executable, str(ROOT / "tools" / "shots.py")],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              env={**os.environ, "SHOTS_PORT": str(PORT), "NFB_NO_OUTPUTS": "1",
                                   # Belt and braces with the suppression inside shots.py: if a future
                                   # Flet reaches for the browser by another route, send it to a command
                                   # that does nothing rather than at the operator's screen.
                                   "BROWSER": "/usr/bin/true"})
    written = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=2)
            for attempt in range(40):         # the server binds when it binds
                try:
                    page.goto(f"http://localhost:{PORT}/launcher", timeout=2000)
                    break
                except Exception:
                    if server.poll() is not None:
                        sys.exit(f"the shot server died:\n{server.stderr.read().decode()}")
                    time.sleep(0.5)
            else:
                sys.exit("the shot server never answered")

            for shot in shots:
                route = route_of(shot)
                if args.tab is not None and route != "/launcher":
                    route += f"/{args.tab}"
                if args.help_title and route != "/launcher":
                    route += f"/help:{quote(args.help_title)}"
                page.goto(f"http://localhost:{PORT}{route}")
                # Readiness is signalled by the page, not slept on — see shots.py.
                page.wait_for_function(f"document.title === 'ready:{route}'", timeout=60000)
                page.wait_for_timeout(250)    # one frame for the canvas to settle after the signal
                if args.help_title:
                    # A dialog drawn over the screen is drawn before the symbol font arrives, so the
                    # arrows behind it come out as boxes unless the shot waits for it.
                    page.wait_for_timeout(1500)
                name = (route.split("/help:")[0].strip("/").replace("/", "-")
                        + ("-help" if args.help_title else "") + f"-{args.width}.png")
                page.screenshot(path=str(out / name), full_page=False)
                written.append(name)
            browser.close()
    finally:
        server.terminate()

    print(f"{len(written)} shots in {out}")
    for name in written:
        print(f"  {out / name}")


if __name__ == "__main__":
    main()
