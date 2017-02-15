import numpy as np
import networkx as nx


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


def convert_to_adj_matrix(matrix):
    g = nx.Graph(matrix)
    adj = nx.adjacency_matrix(g)
