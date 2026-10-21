from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_cv_basics(alpha_lasso_cv, alpha_ridge_cv, mse_lasso, mse_ridge, mse_ols, selected_cv, alphas):
    import numpy as np
    assert isinstance(alpha_lasso_cv, float) and isinstance(alpha_ridge_cv, float)
    assert np.min(alphas) <= alpha_lasso_cv <= np.max(alphas) and np.min(alphas) <= alpha_ridge_cv <= np.max(alphas)
    assert min(mse_lasso, mse_ridge) < 1.5 * mse_ols, 'a CV-tuned penalty should not be much worse than OLS'
    assert isinstance(selected_cv, list) and 1 <= len(selected_cv) <= 10
    assert {0, 1} <= set(selected_cv) or {3, 4} <= set(selected_cv) or {0, 4} <= set(selected_cv) or ({1, 3} <= set(selected_cv)), 'the strong true features (or their copies) must survive'

