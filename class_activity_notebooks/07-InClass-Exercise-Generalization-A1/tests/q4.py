from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 3

@test_case(points=None, hidden=False)
def test_learning_curve_shapes(lc_train, lc_val, N_GRID, NOISE_SD):
    import numpy as np
    assert set(lc_train) == {1, 12} and set(lc_val) == {1, 12}, 'keys must be the degrees 1 and 12'
    for d in (1, 12):
        assert np.asarray(lc_train[d]).shape == (len(N_GRID),) and np.asarray(lc_val[d]).shape == (len(N_GRID),)
    assert lc_train[12][0] < 1e-06, 'a degree-12 polynomial through 10 points should have (essentially) zero training error'
    assert lc_val[1][-1] > 2 * NOISE_SD ** 2, "the straight line's validation error should stay well above the noise floor"
    gap12 = np.asarray(lc_val[12]) - np.asarray(lc_train[12])
    assert gap12[-1] < gap12[0] / 10, 'the degree-12 train/val gap should shrink by more than 10x from n=10 to n=200'

