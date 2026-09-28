from otter.test_files import test_case

OK_FORMAT = False

name = "q6"
points = 3

@test_case(points=None, hidden=False)
def test_cov_sweep(bic_by_cov, ari_by_cov, best_cov):
    assert set(bic_by_cov) == {'spherical', 'diag', 'tied', 'full'}, 'bic_by_cov needs all four covariance types'
    assert set(ari_by_cov) == {'spherical', 'diag', 'tied', 'full'}, 'ari_by_cov needs all four covariance types'
    assert all((-1.0 <= v <= 1.0 for v in ari_by_cov.values()))
    assert best_cov in bic_by_cov and bic_by_cov[best_cov] == min(bic_by_cov.values()), 'best_cov must be the type with the lowest BIC'

