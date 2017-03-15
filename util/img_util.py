from util.math_func import matrix_min, area_in_cycle
from copy import deepcopy
import numpy as np
import pandas as pd
from itertools import product
from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt
from scipy import ndimage

max_x, max_y = 480, 640

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
        # fig, ax = plt.subplots(1, 4)
        fontdict = {'size': 30, 'name': 'Times New Roman'}
        # ax[0].set_xlabel('a. Roberts', fontdict)
        fig, ax = plt.subplots()
        ax.imshow(edges['roberts'], cmap=plt.cm.gray)
        # ax[0].imshow(edges['roberts'], cmap=plt.cm.gray)
        # ax[1].set_xlabel('b. Sobel', fontdict)
        fig, ax = plt.subplots()
        ax.imshow(edges['sobel'], cmap=plt.cm.gray)
        # ax[1].imshow(edges['sobel'], cmap=plt.cm.gray)
        # ax[2].set_xlabel('c. Scharr', fontdict)
        fig, ax = plt.subplots()
        ax.imshow(edges['scharr'], cmap=plt.cm.gray)
        # ax[2].imshow(edges['scharr'], cmap=plt.cm.gray)
        # ax[3].set_xlabel('d. Prewitt', fontdict)
        fig, ax = plt.subplots()
        ax.imshow(edges['prewitt'], cmap=plt.cm.gray)
        # ax[3].imshow(edges['prewitt'], cmap=plt.cm.gray)
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


def neighbours(row, col):
    vertices = [encode_vertex(min(row + 1, max_x - 1), col),
                encode_vertex(max(0, row - 1), col),
                encode_vertex(row, min(col + 1, max_y - 1)),
                encode_vertex(row, max(0, col - 1))]
    return vertices


def single_sweep(row, col, srange):
    vertices = [encode_vertex(min(row + srange, max_x - 1),
                              min(col + srange, max_y - 1)),
                encode_vertex(max(0, row - srange),
                              max(0, col - srange)),
                encode_vertex(max(0, row - srange),
                              min(col + srange, max_y - 1)),
                encode_vertex(min(row + srange, max_x - 1),
                              max(0, col - srange)),
                encode_vertex(min(row + srange, max_x - 1), col),
                encode_vertex(max(0, row - srange), col),
                encode_vertex(row, min(col + srange, max_y - 1)),
                encode_vertex(row, max(0, col - srange))]
    return vertices


def expand_search_range(vertex, sweep_range=2):
    adjacent = []
    row, col = decode_vertex(vertex)
    for i in range(sweep_range):
        sweep = single_sweep(row, col, i + 1)
        adjacent.extend(sweep)
    return adjacent


def cycle_traverse(df, row, col):
    stack = [(row, col)]
    path = []
    path_checker = {}
    paths, cycles = [], []
    prev_map, cycle_map = {}, {}
    while len(stack) > 0:
        r, c = stack.pop()
        cur = encode_vertex(r, c)  # O(1)
        adj = expand_search_range(cur, sweep_range=7)  # O(8) = O(1)
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
            break  # Super hacky, uses fact sweeper starts closest
        path.append(cur)
        path_checker[cur] = 0
    return paths, cycles, prev_map


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


def check_empty(matrix):
    for row in matrix:
        for x in row:
            if x > 0.15:
                return False
    return True


def run_full_algorithm(img='d:/pictures/fydp/nudes/bwcleanbox.jpg'):
    """

    :param img:
    :return:
    :usage:
        >>> img = 'd:/pictures/fydp/nudes2/2.jpg'
        >>> fig, ax = plt.subplots(); ax.imshow(minned, cmap=plt.cm.gray)
    """
    empty = 'd:/pictures/fydp/nudes1/image1.jpg'
    empty_edges = get_all_edges(empty)
    img_edges = get_all_edges(img)
    # tmp = imread(img, as_grey=True) - imread(empty, as_grey=True)
    # img_edges = get_all_edges(tmp)
    minned = matrix_min(img_edges.values()) - matrix_min(empty_edges.values())
    minned[minned < 0] = 0
    # minned = matrix_min(edges.values())

    # cutoff pixels based on non-zero median
    cutoff = 0
    minned = ndimage.median_filter(minned, 4)
    for x in range(3):
        cutoff = np.nanmedian(np.nanmedian(np.where(minned != 0,
                                                    minned,
                                                    np.NaN)))
        minned[minned < cutoff] = 0
    minned[minned >= cutoff] = 1
    all_paths, all_cycles = find_all_cycles(minned)
    areas = pd.Series(get_areas(all_paths, all_cycles))
