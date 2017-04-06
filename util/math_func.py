"""
 Semaphore - Image Processing Algorithm
 Image Processing Algorithm component of Semaphore
 See https://shlchoi.github.io/semaphore/ for more information about Semaphore

 math_func.py
 Copyright (C) 2017 Matthew Chum, Samson H. Choi

 See https://github.com/mattcwc/semaphore-raspi/blob/master/LICENSE for license information
 """

from itertools import tee


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
    for v, vertex in enumerate(vertices):
        if vertex not in visited:
            cycle_id_pairs, used_edges, path = [], [], [vertex]
            iter_dfs_explore(path, edges, cycle_id_pairs, used_edges)
            for start, end in cycle_id_pairs:
                cycle_path = path[start:end]
                cycle_path.append(path[start])
                cycles.append(cycle_path)
            if used_edges is not None:
                visited_edges.extend(used_edges)
            if path is not None:
                cycles.append(path)
                visited = visited.union(set(path))
    return cycles


def dfs_explore(path, edges, cycle_idx_pairs, used_edges):
    curr_vertex = path[-1]
    for edge in edges:
        if curr_vertex in edge and edge not in used_edges:
            v1, v2 = edge
            next_vertex = v2 if v1 == curr_vertex else v1
            used_edges.append(edge)  # Don't re-use the edge
            if len(path) > 2 and next_vertex in path:
                # Cycle is found, record the vertices
                start_vertex = path.index(next_vertex)
                end_vertex = len(path)
                cycle_idx_pairs.append((start_vertex, end_vertex))
            if next_vertex not in path:
                # Add and keep searching
                path.append(next_vertex)
                dfs_explore(path, edges, cycle_idx_pairs, used_edges)


def iter_dfs_explore(path, edges, cycle_idx_pairs, used_edges):
    stack = [path[-1]]
    while len(stack) > 0:
        curr_vertex = stack.pop()
        for edge in edges:
            if curr_vertex in edge and edge not in used_edges:
                v1, v2 = edge
                next_vertex = v2 if v1 == curr_vertex else v1
                used_edges.append(edge)  # Don't re-use the edge
                if len(path) > 2 and next_vertex in path:
                    # Cycle is found, record the vertices
                    start_vertex = path.index(next_vertex)
                    end_vertex = len(path)
                    cycle_idx_pairs.append((start_vertex, end_vertex))
                if next_vertex not in path:
                    # Add and keep searching
                    path.append(next_vertex)
                    stack.append(next_vertex)
