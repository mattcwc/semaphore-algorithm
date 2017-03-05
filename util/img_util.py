from util.math_func import matrix_min, area_in_cycle
from util.server_util import time_run
from copy import deepcopy
import numpy as np
import pandas as pd
from itertools import product
from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt

max_x, max_y = 640, 480

EDGE_METHODS = {'roberts': roberts,
                'sobel': sobel,
                'scharr': scharr,
                # 'canny': canny,
                'prewitt': prewitt}


def get_edges(img, method='roberts', *edge_args, **edge_kwargs):
    """

    :param img: path to image or 2d numpy array representation of
                grey-scale image
    :param method: Method of edge detection
    :return:
    """
    if method not in EDGE_METHODS:
        raise NotImplementedError('Edge filtering method <{}> not implemented')
    if isinstance(img, str):
        img = imread(img, as_grey=True)
    edge = EDGE_METHODS.get(method)(img, *edge_args, **edge_kwargs)
    return edge


def get_all_edges(img, plot=False):
    """

    :param img:
    :param plot:
    :return:
    :usage:
        >>> img = 'd:/pictures/fydp/q30_640x480.jpg'
        >>> edges = get_all_edges(img, plot=True)
        >>> minned = matrix_min(edges.values())
        >>> fig, ax = plt.subplots(); ax.imshow(minned, cmap=plt.cm.gray)
    """
    if isinstance(img, str):
        img = imread(img, as_grey=True)
    edges = {}
    for i, m in enumerate(EDGE_METHODS.keys()):
        edges[m] = get_edges(img, m)
    if plot:
        fig, ax = plt.subplots(1, 4)
        fontdict = {'size': 30, 'name': 'Times New Roman'}
        ax[0].set_xlabel('a. Roberts', fontdict)
        ax[0].imshow(edges['roberts'], cmap=plt.cm.gray)
        ax[1].set_xlabel('b. Sobel', fontdict)
        ax[1].imshow(edges['sobel'], cmap=plt.cm.gray)
        ax[2].set_xlabel('c. Scharr', fontdict)
        ax[2].imshow(edges['scharr'], cmap=plt.cm.gray)
        ax[3].set_xlabel('d. Prewitt', fontdict)
        ax[3].imshow(edges['prewitt'], cmap=plt.cm.gray)
    return edges


def clean_edges(matrix):
    """

    :param matrix: 2d array
    :return: aggregated 2d array of edges
    """
    # Remove outliers
    ret = deepcopy(matrix)
    height, width = matrix.shape
    base = 17
    x, y = base, base
    while x < height:
        while y < width:
            sub_matrix = matrix[x - base:x, y - base: y]
            if np.mean(sub_matrix) < 0.5:
                ret[x - base:x, y - base: y] = 0
            y += 1
        x += 1
        y = base
    return ret


def two_dimension_numpy_intersect(m1, m2):
    """
    Code snippet of quick 2d numpy array intersection which is faster than
    Python set intersection
    :return:
    """
    nrows, ncols = m1.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [m1.dtype]}
    intersection = np.intersect1d(m1.view(dtype), m2.view(dtype))
    intersection = intersection.view(m1.dtype).reshape(-1, ncols)
    return intersection


def encode_vertex(row, col):
    return row * 10000 + col


def decode_vertex(number):
    row = number / 10000
    col = number - row * 10000
    return row, col


# def adjacent_vertices(row, col):
#     # Only 2 because shoddy attempt at makign it underected
#     vertices = [(min(row + 1, max_x), col),
#                 (row, min(col + 1, max_y))]
#     return np.array(vertices)
#
#
# def single_convert(df, row, col):
#     if df.iloc[row, col] != 0:
#         vertex = encode_vertex(row, col)
#         edges = []
#         for adj_r, adj_c in adjacent_vertices(row, col):
#             v = encode_vertex(adj_r, adj_c)
#             if df.iloc[adj_r, adj_c] != 0:
#                 edges.append((vertex, v))
#         return vertex, edges
#     return None, None
#
#
# @time_run
# def convert_adjacency_list(matrix):
#     vertices, adj_edges = [], []
#     tmp = pd.DataFrame(matrix)
#     rows, cols = matrix.shape
#     for r, c in product(list(range(rows)), list(range(cols))):
#         vertex, edges = single_convert(tmp, r, c)
#         if vertex is not None:
#             vertices.append(vertex)
#             adj_edges.extend(edges)
#     return vertices, adj_edges


def neighbours(row, col):
    vertices = [encode_vertex(min(row + 1, max_x), col),
                encode_vertex(max(0, row - 1), col),
                encode_vertex(row, min(col + 1, max_y)),
                encode_vertex(row, max(0, col - 1))]
    return vertices


def cycle_traverse(df, row, col):
    stack = [(row, col)]
    path = []
    path_checker = {}
    paths, cycles = [], []
    prev_map, cycle_map = {}, {}
    while len(stack) > 0:
        r, c = stack.pop()
        cur = encode_vertex(r, c)  # O(1)
        adj = neighbours(r, c)  # O(4) = O(1)
        prev = path[-1] if len(path) > 1 else -1  # O(1)
        if prev > 0 and prev not in adj:  # Total O(n) worst case
            # Save the old cycles
            paths.append(path)
            cycles.append(cycle_map)

            # Truncate back to where we came from, reset cycles
            path = path[:path.index(prev) + 1]  # Worst case O(len(path))
            path_checker = {k: 0 for k in path}  # Worst case O(len(path))
            cycle_map = {}
        for nxt in adj:
            nxt_row, nxt_col = decode_vertex(nxt)  # O(1)
            if nxt == prev or df.iloc[nxt_row, nxt_col] == 0:  # O(1)
                continue  # Ignore unwanted cells
            if path_checker.get(nxt, None) is None:  # O(1)
                # No cycle, neighbour is another node, put it on stack to visit
                prev_map[nxt] = cur
                if (nxt_row, nxt_col) not in stack:
                    stack.append((nxt_row, nxt_col))
            else:
                # Cycle exists because it was previously passed through
                n_idx, c_idx = path.index(nxt), len(path)
                if cycle_map.get(n_idx, None) is None:  # O(1)
                    cycle_map[n_idx] = []
                cycle_map[n_idx].append(c_idx)  # O(n)
        path.append(cur)
        path_checker[cur] = 0
    return paths, cycles, prev_map


@time_run
def find_all_cycles(matrix):
    df = pd.DataFrame(matrix)
    rows, cols = matrix.shape
    all_paths = []
    all_cycles = []
    ignore = {}
    for r, c in product(list(range(rows)), list(range(cols))):
        find_flag = ignore.get(encode_vertex(r, c), None) is None
        if df.iloc[r, c] != 0 and find_flag:
            paths, cycles, visited = cycle_traverse(df, r, c)
            all_paths.extend(paths)
            all_cycles.extend(cycles)
            ignore.update(visited)
    return all_paths, all_cycles


@time_run
def get_areas(all_paths, all_cycles):
    areas = []
    for p, c in zip(all_paths, all_cycles):  # O(len(all_paths))
        for s_idx, e_list in c.iteritems():  # O(unique start nodes)
            for e_idx in e_list:  # O(unique start to end)
                cycle_path = p[s_idx:e_idx]  # O(k)
                cycle_path.append(p[s_idx])  # O(1)
                decoded = [decode_vertex(x) for x in cycle_path]  # O(cycle len)
                areas.append(area_in_cycle(decoded))  # O(cycle len)
    return areas


def run_full_algorithm(img='d:/pictures/fydp/q30_640x480.jpg'):
    edges = get_all_edges(img)
    minned = matrix_min(edges.values())

    # cutoff pixels based on non-zero median
    cutoff = np.nanmedian(np.nanmedian(np.where(minned != 0, minned, np.NaN)))
    minned[minned < cutoff] = 0
    all_paths, all_cycles = find_all_cycles(minned)
    areas = get_areas(all_paths, all_cycles)
    # graph = convert_adjacency_list(minned)  # Also encodes vertices
    # cycles = cycle_finder(*graph)

