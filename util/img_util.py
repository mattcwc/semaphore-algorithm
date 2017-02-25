from util.math_func import matrix_min, cycle_finder, area_in_cycle
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
        >>> img = 'd:/pictures/fydp/7_letters.jpg'
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
        fig, ax = plt.subplots(2, 2)
        ax[0][0].set_title('roberts')
        ax[0][0].imshow(edges['roberts'], cmap=plt.cm.gray)
        ax[0][1].set_title('sobel')
        ax[0][1].imshow(edges['sobel'], cmap=plt.cm.gray)
        ax[1][0].set_title('scharr')
        ax[1][0].imshow(edges['scharr'], cmap=plt.cm.gray)
        ax[1][1].set_title('prewitt')
        ax[1][1].imshow(edges['prewitt'], cmap=plt.cm.gray)
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
    col = number - row
    return row, col


def adjacent_vertices(row, col):
    # Only 2 because shoddy attempt at makign it underected
    vertices = [(min(row + 1, max_x), col),
                (row, min(col + 1, max_y))]
    return np.array(vertices)


def single_convert(df, row, col):
    if df.iloc[row, col] != 0:
        vertex = encode_vertex(row, col)
        edges = []
        for adj_r, adj_c in adjacent_vertices(row, col):
            v = encode_vertex(adj_r, adj_c)
            if df.iloc[adj_r, adj_c] != 0:
                edges.append((vertex, v))
        return vertex, edges
    return None, None


@time_run
def convert_adjacency_list(matrix):
    vertices, adj_edges = [], []
    tmp = pd.DataFrame(matrix)
    rows, cols = matrix.shape
    for r, c in product(list(range(rows)), list(range(cols))):
        vertex, edges = single_convert(tmp, r, c)
        if vertex is not None:
            vertices.append(vertex)
            adj_edges.extend(edges)
    return vertices, adj_edges


def run_full_algorithm():
    img = 'd:/pictures/fydp/q30_640x480.jpg'
    edges = get_all_edges(img)
    minned = matrix_min(edges.values())

    # cutoff pixels based on non-zero median
    cutoff = np.nanmedian(np.nanmedian(np.where(minned != 0, minned, np.NaN)))
    minned[minned < cutoff] = 0
    graph = convert_adjacency_list(minned)  # Also encodes vertices
    cycles = cycle_finder(*graph)

    # Decoding vertices and getting areas
    areas = []
    for c in cycles:
        cycle = [decode_vertex(v) for v in c]
        areas.append(area_in_cycle(cycle))
