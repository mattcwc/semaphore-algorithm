from util.math_func import matrix_min
from copy import deepcopy
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt
from matplotlib import pyplot as plt


max_x, max_y = 3264, 2448

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
        >>> fig, ax = plt.subplots(); ax.imshow(cleaned, cmap=plt.cm.gray)
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


def get_graph_vertices(matrix):
    tmp = pd.DataFrame(matrix)
    work = tmp[tmp > 0]
    vertices = []
    for c in work.columns:
        for i in work[c].dropna().index:
            vertices.append((c, i))
    return np.array(vertices)


def get_surrounding_vertices(v):
    x, y = v
    vertices = [(max(x - 1, 0), y),
                (min(x + 1, max_x), y),
                (x, max(y - 1, 0)),
                (x, min(y + 1, max_y))]
    return np.array(vertices)


def get_graph_edges(vertices):
    graph_edges = pd.DataFrame(columns=[0, 1], dtype=int)
    tmp = pd.DataFrame(vertices)
    for i, v in enumerate(vertices):
        surrounding = get_surrounding_vertices(v)
        intersection = two_dimension_numpy_intersect(surrounding, vertices)
        for n in intersection:
            j = tmp[(tmp[0] == n[0]) & (tmp[1] == n[1])].index[0]
            check = graph_edges[(graph_edges[0] == j) & (graph_edges[1] == i)]
            if len(check) == 0:
                graph_edges.append(pd.Series({0: i, 1: j}), ignore_index=True)
    return graph_edges.as_matrix()


def convert_adjacency_list(matrix):
    graph_vertices = get_graph_vertices(matrix)
    graph_edges = get_graph_edges(graph_vertices)  # This part takes 4 hours
    return graph_vertices, graph_edges


def main():
    img = 'd:/pictures/fydp/7_letters.jpg'
    edges = get_all_edges(img)
    minned = matrix_min(edges.values())

    # Pixels to 0 or 1 based on intensity
    minned[minned > 0.015] = 1
    minned[minned <= 0.015] = 0
    cleaned = clean_edges(minned)  # Based on edge confidence
    graph = convert_adjacency_list(cleaned)
