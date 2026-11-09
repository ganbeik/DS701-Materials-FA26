from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_svd_shapes(lambdas_svd, Y_svd, V2_svd, X):
    import numpy as np
    assert np.asarray(lambdas_svd).shape == (X.shape[1],), 'lambdas_svd must have one entry per feature'
    assert np.asarray(Y_svd).shape == (X.shape[0], 2), 'Y_svd must be (m, 2)'
    assert np.asarray(V2_svd).shape == (X.shape[1], 2), 'V2_svd must be (n, 2)'

@test_case(points=None, hidden=False)
def test_svd_matches_pca(lambdas_svd, Y_svd, V2_svd, X_pca, pca_full):
    import numpy as np
    assert np.allclose(lambdas_svd, pca_full.explained_variance_), 'lambda_i should equal sigma_i^2 / (m - 1)'
    assert np.allclose(np.abs(Y_svd), np.abs(X_pca)), "U' Sigma' must equal the PCA scores up to sign"
    assert np.allclose(np.abs(V2_svd), np.abs(pca_full.components_[:2].T)), 'columns of V must be the principal directions up to sign'

