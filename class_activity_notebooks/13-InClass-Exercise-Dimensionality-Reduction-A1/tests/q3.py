from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_recon_shape(X_recon, recon_frac, X):
    import numpy as np
    assert np.asarray(X_recon).shape == X.shape, 'X_recon must have the same shape as X'
    assert isinstance(recon_frac, float), 'recon_frac must be a float'
    assert 0.0 < recon_frac <= 0.1, 'with >= 90% variance retained you should lose at most 10% of the sum of squares'

@test_case(points=None, hidden=False)
def test_recon_matches_evr(recon_frac, cum_evr, n_components_90):
    import numpy as np
    assert np.isclose(recon_frac, 1 - np.asarray(cum_evr)[n_components_90 - 1], atol=1e-06), 'the lost fraction should equal 1 minus the cumulative explained variance at d'

