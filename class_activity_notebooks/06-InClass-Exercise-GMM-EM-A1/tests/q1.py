from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 3

@test_case(points=None, hidden=False)
def test_e_step_toy(e_step):
    import numpy as np
    g = e_step(np.array([-1.0, 0.0, 2.5]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    g = np.asarray(g)
    assert g.shape == (3, 2), f'expected shape (3, 2), got {g.shape}'
    assert np.allclose(g, 0.5), 'identical components must split every point 50/50'
    g = e_step(np.array([0.0, 3.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    g = np.asarray(g)
    assert np.allclose(g.sum(axis=1), 1.0), 'each row of gamma must sum to 1'
    assert np.allclose(g[0], [0.5, 0.5]), 'the midpoint of two symmetric components is 50/50'
    assert g[1, 1] > 0.99, 'a point far to the right must belong to the right component'

