from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 3

@test_case(points=None, hidden=False)
def test_resp_basics(km2, gm2, resp, max_resp, n_uncertain, frac_uncertain, ari_km_gmm, X2):
    import numpy as np
    assert hasattr(km2, 'labels_') and hasattr(gm2, 'means_'), 'fit km2 and gm2 first'
    resp = np.asarray(resp)
    assert resp.shape == (len(X2), 3), f'resp must be (n, 3), got {resp.shape}'
    assert np.allclose(resp.sum(axis=1), 1.0), 'each row of resp must sum to 1'
    assert np.allclose(np.asarray(max_resp), resp.max(axis=1))
    assert 0 < n_uncertain < len(X2) and np.isclose(frac_uncertain, n_uncertain / len(X2))
    assert -0.5 <= ari_km_gmm <= 1.0

