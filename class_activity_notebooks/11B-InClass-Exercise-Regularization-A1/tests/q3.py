from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_paths_shape(ridge_path, lasso_path, n_nonzero_lasso, selected_at_ref, alphas, FEATURES):
    import numpy as np
    assert np.asarray(ridge_path).shape == (len(alphas), len(FEATURES))
    assert np.asarray(lasso_path).shape == (len(alphas), len(FEATURES))
    assert len(n_nonzero_lasso) == len(alphas)
    assert all(np.diff(n_nonzero_lasso) <= 0) or n_nonzero_lasso[0] >= n_nonzero_lasso[-1], 'more penalty -> fewer nonzero coefficients'
    assert (np.abs(ridge_path) > 1e-08).all(), 'ridge never sets a coefficient exactly to zero'
    assert n_nonzero_lasso[-1] < n_nonzero_lasso[0], 'lasso should zero out coefficients as alpha grows'
    assert isinstance(selected_at_ref, list) and selected_at_ref == sorted(selected_at_ref)

