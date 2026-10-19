from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 2

@test_case(points=None, hidden=False)
def test_floor_and_auc_basics(clf_loan, p_test, acc_floor, acc_05, auc_test, y_loan_test):
    import numpy as np
    p = np.asarray(p_test)
    assert p.shape == y_loan_test.shape, 'one probability per held-out applicant'
    assert p.min() >= 0 and p.max() <= 1, 'p_test must be probabilities -- use predict_proba(...)[:, 1]'
    assert isinstance(acc_floor, float) and isinstance(acc_05, float) and isinstance(auc_test, float)
    assert abs(acc_floor - np.mean(y_loan_test == 1)) < 1e-12, 'approving everyone is right exactly when the applicant repays'
    assert acc_floor > 0.85, 'about nine applicants in ten repay'
    assert 0.5 < auc_test <= 1.0

