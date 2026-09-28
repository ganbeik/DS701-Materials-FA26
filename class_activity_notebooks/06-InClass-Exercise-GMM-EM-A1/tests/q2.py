from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 3

@test_case(points=None, hidden=False)
def test_m_step_hard_labels(m_step):
    import numpy as np
    x_t = np.array([0.0, 2.0, 10.0, 12.0, 14.0])
    g_t = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=float)
    mu, sigma, pi = m_step(x_t, g_t)
    assert np.allclose(mu, [1.0, 12.0]), f'means should be [1, 12], got {mu}'
    assert np.allclose(sigma, [1.0, np.sqrt(8 / 3)]), f'stds should be [1, sqrt(8/3)], got {sigma}'
    assert np.allclose(pi, [0.4, 0.6]), f'mixing weights should be [0.4, 0.6], got {pi}'
    assert np.isclose(np.sum(pi), 1.0)

