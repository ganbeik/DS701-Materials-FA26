from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 4

@test_case(points=None, hidden=False)
def test_run_em_shape(mu_hat, sigma_hat, pi_hat, loglik_hist, log_likelihood, x):
    import numpy as np
    assert np.asarray(mu_hat).shape == (2,) and np.asarray(sigma_hat).shape == (2,) and (np.asarray(pi_hat).shape == (2,))
    assert np.isclose(np.sum(pi_hat), 1.0), 'mixing weights must sum to 1'
    assert np.all(np.asarray(sigma_hat) > 0), 'standard deviations must be positive'
    assert isinstance(loglik_hist, list) and len(loglik_hist) >= 3, 'loglik_hist must be a list with the initial value plus one entry per iteration'
    assert isinstance(log_likelihood(x, np.asarray(mu_hat), np.asarray(sigma_hat), np.asarray(pi_hat)), float)
    diffs = np.diff(np.asarray(loglik_hist))
    assert np.all(diffs >= -1e-08), 'the log-likelihood must never decrease from one EM iteration to the next'

