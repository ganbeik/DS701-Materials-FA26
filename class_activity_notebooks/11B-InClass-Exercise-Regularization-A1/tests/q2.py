from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_bootstrap_basics(boot_coefs, boot_std, swing_ratio, pair0_sum_std, N_BOOT):
    import numpy as np
    assert np.asarray(boot_coefs).shape == (N_BOOT, 10), 'one row of 10 coefficients per resample'
    assert np.asarray(boot_std).shape == (10,) and np.all(np.asarray(boot_std) >= 0)
    assert isinstance(swing_ratio, float) and isinstance(pair0_sum_std, float)
    assert swing_ratio > 1, 'the attributes with a near-twin should swing more than the others'
    assert pair0_sum_std < boot_std[0], "the pair's total should be steadier than either coefficient alone"

