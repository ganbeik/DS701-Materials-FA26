from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 3

@test_case(points=None, hidden=False)
def test_predict_cf_shape(predict_cf, pred_cf, rmse_cf, holdout, K_CF, user_mean):
    import numpy as np
    assert K_CF == 10
    assert np.asarray(pred_cf).shape == (len(holdout),), 'one prediction per held-out pair'
    assert np.all(np.isfinite(pred_cf))
    p = predict_cf(int(holdout[0, 0]), int(holdout[0, 1]), K_CF)
    assert abs(p - pred_cf[0]) < 1e-09, 'pred_cf must come from predict_cf with K_CF'
    assert 0.3 < rmse_cf < 1.5

