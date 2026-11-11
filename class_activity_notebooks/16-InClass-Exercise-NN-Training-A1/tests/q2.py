from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_grid_shape(results, curves, mean_acc, better_solver, SOLVERS, LRS, BUDGET):
    import numpy as np
    keys = {(s, lr) for s in SOLVERS for lr in LRS}
    assert set(results) == keys, 'results must have one entry per (solver, lr) pair'
    assert set(curves) == keys
    assert all((0.5 <= v <= 1.0 for v in results.values()))
    assert all((1 <= len(np.asarray(c)) <= BUDGET for c in curves.values())), 'each model gets at most BUDGET epochs'
    assert set(mean_acc) == set(SOLVERS)
    assert better_solver == max(mean_acc, key=mean_acc.get), 'better_solver must be the solver with the higher mean_acc'

