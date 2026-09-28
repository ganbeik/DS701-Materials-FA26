from otter.test_files import test_case

OK_FORMAT = False

name = "q7"
points = 2

@test_case(points=None, hidden=False)
def test_half_point(idx_half, resp_half, km_label_half, resp, X2):
    import numpy as np
    assert 0 <= idx_half < len(X2)
    resp_half = np.asarray(resp_half)
    assert resp_half.shape == (3,) and np.isclose(resp_half.sum(), 1.0)
    assert np.allclose(resp_half, np.asarray(resp)[idx_half]), 'resp_half must be row idx_half of resp'
    assert resp_half.max() < 0.6, 'the chosen point should have no component above 0.6'
    assert km_label_half in (0, 1, 2)

