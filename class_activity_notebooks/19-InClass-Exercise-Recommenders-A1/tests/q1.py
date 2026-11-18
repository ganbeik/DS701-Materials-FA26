from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 2

@test_case(points=None, hidden=False)
def test_baselines_shape(sparsity, mu, item_mean, rmse_mu, rmse_item_mean, n_items):
    import numpy as np
    assert 0.6 < sparsity < 0.8, 'about 70% of the matrix should be missing'
    assert 1.0 < mu < 5.0
    assert np.asarray(item_mean).shape == (n_items,) and (not np.isnan(item_mean).any())
    assert rmse_mu > 0 and rmse_item_mean > 0

