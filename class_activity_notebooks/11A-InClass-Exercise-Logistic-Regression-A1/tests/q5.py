from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 2

@test_case(points=None, hidden=False)
def test_threshold_sweep_basics(prec_t, rec_t, f1_t, t_best_f1, thresholds):
    import numpy as np
    for arr in (prec_t, rec_t, f1_t):
        assert np.asarray(arr).shape == thresholds.shape, 'one value per threshold'
    assert np.all(np.diff(rec_t) <= 1e-12), 'raising the threshold can only approve fewer repayers: recall never rises'
    assert rec_t[0] > 0.99, 'at t = 0.05 nearly every repayer is approved'
    assert isinstance(t_best_f1, float) and np.any(np.isclose(thresholds, t_best_f1))

