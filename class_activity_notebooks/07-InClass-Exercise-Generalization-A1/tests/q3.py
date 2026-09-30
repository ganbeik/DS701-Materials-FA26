from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 2

@test_case(points=None, hidden=False)
def test_bias_variance_shape(bias2, variance):
    assert set(bias2) == {1, 9} and set(variance) == {1, 9}, 'keys must be the degrees 1 and 9'
    assert all((v >= 0 for v in bias2.values())) and all((v >= 0 for v in variance.values()))
    assert bias2[1] > bias2[9], 'the straight line should have the larger bias'
    assert variance[9] > variance[1], 'the degree-9 polynomial should have the larger variance'

