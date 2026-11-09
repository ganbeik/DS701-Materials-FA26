from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 4

@test_case(points=None, hidden=False)
def test_sets_format(valid_set, invalid_set, DAG_NAMES, DAGS):
    import networkx as nx
    assert set(valid_set) == set(DAG_NAMES) and set(invalid_set) == set(DAG_NAMES), 'one entry per diagram'
    for name in DAG_NAMES:
        nodes = set(nx.DiGraph(DAGS[name]['edges']).nodes)
        for label, S in (('valid_set', valid_set[name]), ('invalid_set', invalid_set[name])):
            S = set(S)
            assert S <= nodes, f'{label}[{name!r}] uses nodes not in the diagram: {S - nodes}'
            assert not S & {'X', 'Y'}, f'{label}[{name!r}] must not contain X or Y'
        assert set(valid_set[name]) != set(invalid_set[name]), f'{name}: valid and invalid sets must differ'

