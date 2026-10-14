from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_diagnostics_toy(diagnostics):
    import numpy as np
    X_t = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    d = diagnostics(X_t, np.array([1.0, 3.0, 5.0]))
    assert set(d) == {'r2', 'curve', 'var_ratio', 'max_leverage', 'i_max_leverage', 'leverage_ratio'}, 'wrong set of keys'
    assert abs(d['r2'] - 1.0) < 1e-09
    assert abs(d['max_leverage'] - 5.0 / 6.0) < 1e-09 and d['i_max_leverage'] in (0, 2)
    assert abs(d['leverage_ratio'] - 5.0 / 6.0 / (2.0 / 3.0)) < 1e-09
    assert isinstance(d['i_max_leverage'], int)

@test_case(points=None, hidden=False)
def test_diag_good_ranges(diag_good, r2_2):
    assert abs(diag_good['r2'] - r2_2) < 1e-06, "diagnostics on X2 must reproduce Part 3's R^2"
    assert abs(diag_good['curve']) < 0.3, 'the diabetes model should show no strong bow'
    assert diag_good['leverage_ratio'] < 6.0

