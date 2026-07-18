from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_units import (cubic_miles, format_inches, format_miles, inches,
                        miles, square_miles)


def test_plot_unit_conversions_accept_scalars_and_arrays():
    assert inches(25.4) == 1.0
    assert miles(1.609344) == 1.0
    assert square_miles(1.609344 ** 2) == 1.0
    assert cubic_miles(1.609344 ** 3) == 1.0
    assert np.allclose(inches(np.asarray([25.4, 50.8])), [1.0, 2.0])


def test_plot_unit_labels_are_compact():
    assert format_inches(1.0) == "0.04"
    assert format_inches(25.4) == "1"
    assert format_miles(5.6) == "3.5"
