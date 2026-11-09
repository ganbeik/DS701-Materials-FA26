from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 4

@test_case(points=None, hidden=False)
def test_backdoor_paths_format(my_backdoor_paths, DAG_NAMES, DAGS):
    import networkx as nx
    assert set(my_backdoor_paths) == set(DAG_NAMES), 'one entry per diagram in DAG_NAMES'
    for name in DAG_NAMES:
        G = nx.DiGraph(DAGS[name]['edges'])
        entries = my_backdoor_paths[name]
        assert isinstance(entries, (list, tuple)), f'{name}: value must be a list of (path, status) pairs'
        for entry in entries:
            assert isinstance(entry, (list, tuple)) and len(entry) == 2, f'{name}: each entry is (path, status)'
            path, status = entry
            path = tuple(path)
            assert status in ('open', 'blocked'), f"{name}: status must be 'open' or 'blocked', got {status!r}"
            assert path[0] == 'X' and path[-1] == 'Y', f'{name}: paths run from X to Y, got {path}'
            assert len(set(path)) == len(path), f'{name}: a path visits each node once, got {path}'
            assert all((n in G.nodes for n in path)), f'{name}: unknown node in {path}'
            for a, b in zip(path, path[1:]):
                assert G.has_edge(a, b) or G.has_edge(b, a), f'{name}: {a}-{b} is not an edge in {path}'
            assert G.has_edge(path[1], 'X'), f"{name}: {path} starts with an arrow OUT of X -- that is a causal or 'neither' path, not a backdoor path"

