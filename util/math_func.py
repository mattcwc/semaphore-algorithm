from itertools import tee
from util.server_util import time_run


def matrix_min(matrices):
    assert len(matrices) > 0, 'no matrices to minimize'
    if len(matrices) == 1:
        return matrices[0]
    ret = matrices.pop()
    while len(matrices) > 0:
        m = matrices.pop()
        for a, x in enumerate(m):
            for b, y in enumerate(x):
                if y < ret[a][b]:
                    ret[a][b] = y
    return ret


def area_in_cycle(cycle):
    """
    Taken from www.mathopenref.com/coordpolygonarea.html
    """
    a, b = tee(cycle)
    next(b)
    parts = 0
    for v1, v2 in zip(a, b):
        x1, y1 = v1
        x2, y2 = v2
        parts += x1 * y2 - y1 * x2
    return abs(parts / 2)


@time_run
def cycle_finder(vertices, edges):
    """

    :param vertices:
    :param edges:
    :return:
    :usage:
        >>> import pandas as pd
        >>> graph = pd.read_pickle('D:/pictures/fydp/graph.pkl')
        >>> cycles = cycle_finder(*graph)
    """
    cycles = []
    visited = set()
    visited_edges = []
    for vertex in vertices:
        if vertex not in visited:
            path, used_edges = find_cycle_recursive([vertex], edges,
                                                    visited_edges)
            if used_edges is not None:
                visited_edges.extend(used_edges)
            if path is not None:
                cycles.append(path)
                visited = visited.union(set(path))
    return cycles


def find_cycle_recursive(path, edges, used_edges):
    start_vertex = path[-1]
    for edge in edges:
        if start_vertex in edge and edge not in used_edges:
            v1, v2 = edge
            next_vertex = v2 if v1 == start_vertex else v1
            if len(path) > 2 and next_vertex == path[0]:
                # Cycle found
                n = path.index(min(path))
                path = path[n:] + path[:n]
                used_edges.append(edge)
                return path, used_edges
            if next_vertex not in path:
                path.append(next_vertex)
                used_edges.append(edge)
                return find_cycle_recursive(path, edges, used_edges)
    return None, None
