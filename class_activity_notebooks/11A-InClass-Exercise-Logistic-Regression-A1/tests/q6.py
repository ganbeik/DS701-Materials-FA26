from otter.test_files import test_case

OK_FORMAT = False

name = "q6"
points = 3

@test_case(points=None, hidden=False)
def test_cost_basics(t_star, cost_per_applicant, cost_05, cost_tstar, cost_t, t_best_cost, cost_bal, thresholds):
    import numpy as np
    assert abs(t_star - 0.8) < 1e-12, 't* = C_FP / (C_FP + C_FN)'
    y = np.array([0, 1, 1, 0])
    assert abs(cost_per_applicant(y, np.array([True, False, True, False])) - (30000 + 7500) / 4) < 1e-09, 'one false approval and one false decline among four applicants'
    assert abs(cost_per_applicant(y, np.array([False, True, True, False]))) < 1e-12, 'no mistakes, no cost'
    assert np.asarray(cost_t).shape == thresholds.shape
    assert isinstance(cost_05, float) and isinstance(cost_tstar, float) and isinstance(cost_bal, float)
    assert cost_tstar <= cost_05, 'the cost-optimal threshold should not cost more than 0.5'
    assert np.any(np.isclose(thresholds, t_best_cost))

