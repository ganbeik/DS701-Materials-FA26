from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 2

@test_case(points=None, hidden=False)
def test_gm1_fitted(gm1):
    from sklearn.mixture import GaussianMixture
    assert isinstance(gm1, GaussianMixture), 'gm1 must be a fitted sklearn GaussianMixture'
    assert hasattr(gm1, 'means_'), 'call .fit(...) on gm1'
    assert gm1.means_.shape == (2, 1), 'fit two components on x.reshape(-1, 1)'

