from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 3

@test_case(points=None, hidden=False)
def test_rates_structure(rate_within, rate_overall, share_high, better_within, better_overall, reversal, COUNTS, ARMS, STRATA):
    import numpy as np
    assert set(rate_within) == set(COUNTS), 'rate_within needs one entry per (arm, stratum)'
    assert all((0.0 <= v <= 1.0 for v in rate_within.values())), 'rates are fractions in [0, 1]'
    assert set(rate_overall) == set(ARMS) and set(share_high) == set(ARMS)
    assert all((0.0 <= v <= 1.0 for v in list(rate_overall.values()) + list(share_high.values())))
    assert better_within in ARMS and better_overall in ARMS
    assert isinstance(reversal, (bool, np.bool_))
    for a in ARMS:
        lo = min((rate_within[a, s] for s in STRATA))
        hi = max((rate_within[a, s] for s in STRATA))
        assert lo - 1e-09 <= rate_overall[a] <= hi + 1e-09, f'pooled rate for {a} must lie between its within-stratum rates'

