from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 2

@test_case(points=None, hidden=False)
def test_adjusted_structure(p_stratum, rate_adjusted, better_adjusted, ARMS, STRATA):
    assert set(p_stratum) == set(STRATA) and abs(sum(p_stratum.values()) - 1.0) < 1e-09, 'p_stratum must be a distribution over the strata'
    assert set(rate_adjusted) == set(ARMS)
    assert all((0.0 <= v <= 1.0 for v in rate_adjusted.values()))
    assert better_adjusted in ARMS

