"""Operator console.

Deliberately a separate package: it pulls in Flet and therefore a Flutter runtime, and `nfb`
must stay importable for headless analysis and tests without any of that.

It is also a separate *process* from the feedback path. Sound and shadow are latency-critical and
must never wait on a UI redraw, a file dialog, or a Flutter frame.
"""
