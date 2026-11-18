from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 3

@test_case(points=None, hidden=False)
def test_similarity_shape(user_mean, R_c, S, n_users, n_items, mask):
    import numpy as np
    assert np.asarray(user_mean).shape == (n_users,)
    assert R_c.shape == (n_users, n_items) and (not np.isnan(R_c).any()), 'R_c must have no NaN'
    assert np.allclose(R_c[~mask], 0.0), 'missing ratings must become 0 in R_c'
    assert S.shape == (n_items, n_items)
    assert np.allclose(S, S.T), 'cosine similarity is symmetric'
    assert np.allclose(np.diag(S), 0.0), 'set the diagonal of S to 0'
    assert S.min() >= -1 - 1e-09 and S.max() <= 1 + 1e-09

