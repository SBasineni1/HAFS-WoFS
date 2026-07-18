"""Lightweight unit conversions used by the plotting layer.

Verification calculations and CSV schemas intentionally remain in their
native metric units. Figures convert at render time so scientific thresholds,
configuration values, and existing machine-readable output stay stable.
"""

MM_PER_INCH = 25.4
KM_PER_MILE = 1.609344
KM2_PER_MILE2 = KM_PER_MILE ** 2
KM3_PER_MILE3 = KM_PER_MILE ** 3


def inches(mm):
    if mm is None:
        return float("nan")
    return mm / MM_PER_INCH


def miles(km):
    if km is None:
        return float("nan")
    return km / KM_PER_MILE


def square_miles(km2):
    if km2 is None:
        return float("nan")
    return km2 / KM2_PER_MILE2


def cubic_miles(km3):
    if km3 is None:
        return float("nan")
    return km3 / KM3_PER_MILE3


def format_inches(mm):
    """Compact, readable inch value for thresholds and annotations."""
    return f"{inches(mm):.2f}".rstrip("0").rstrip(".")


def format_miles(km):
    """Compact mile value for discrete scale labels."""
    return f"{miles(km):.1f}".rstrip("0").rstrip(".")
