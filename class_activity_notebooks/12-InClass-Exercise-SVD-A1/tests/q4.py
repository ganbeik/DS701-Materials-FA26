from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_error_energy_shapes(frob_err, frob_err_formula, energy, energy_at, k_99, compression_ratio, KS, n):
    import numpy as np
    for d in (frob_err, frob_err_formula, energy_at, compression_ratio):
        assert list(d) == KS, 'keys must be the values of KS, in order'
        assert all((isinstance(v, float) for v in d.values())), 'store plain floats'
    for k in KS:
        assert abs(frob_err[k] - frob_err_formula[k]) <= 1e-06 * frob_err_formula[k], f'k={k}: direct norm and sqrt(sum of discarded sigma^2) must agree'
    errs = [frob_err[k] for k in KS]
    assert all(np.diff(errs) < 0), 'error must fall as k grows'
    assert energy.shape == (n,) and np.all(np.diff(energy) >= -1e-12) and (abs(energy[-1] - 1.0) < 1e-09)
    assert all((0.0 < energy_at[k] <= 1.0 for k in KS))
    assert isinstance(k_99, int) and 1 <= k_99 <= n
    assert energy[k_99 - 1] >= 0.99 and (k_99 == 1 or energy[k_99 - 2] < 0.99), 'k_99 must be the SMALLEST k reaching 0.99'
    assert compression_ratio[1] > compression_ratio[50] > 1.0

