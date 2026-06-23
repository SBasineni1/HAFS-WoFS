import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ets_score import contingency_scores


def test_hss_perfect_forecast_is_one():
    # Forecast exactly equals obs -> b=c=0 -> HSS = 1.0
    fcst = np.array([10.0, 0.0, 20.0, 0.0])
    obs = np.array([10.0, 0.0, 20.0, 0.0])
    s = contingency_scores(fcst, obs, 5.0)
    assert s["b"] == 0 and s["c"] == 0
    assert abs(s["hss"] - 1.0) < 1e-12


def test_hss_known_table():
    # Build a case with a=2,b=1,c=1,d=2 at threshold 5.
    #   fcst>=5 at idx 0,1,2 ; obs>=5 at idx 0,2,4 (over 6 points)
    fcst = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    obs = np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    s = contingency_scores(fcst, obs, 5.0)
    assert (s["a"], s["b"], s["c"], s["d"]) == (2, 1, 1, 2)
    # HSS = 2(ad-bc)/((a+c)(c+d)+(a+b)(b+d))
    #     = 2(4-1)/((3)(3)+(3)(3)) = 6/18 = 0.3333...
    assert abs(s["hss"] - (6.0 / 18.0)) < 1e-12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
