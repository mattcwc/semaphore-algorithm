from util.img_util import get_all_edges, find_all_cycles, get_areas
from util.math_func import matrix_min
from util.fuzzy_logic import get_objects, find_objects
from scipy import ndimage
import numpy as np
import pandas as pd

from util.cv2_util import find_straight_lines, convert_graph, find_cycles, \
    bresenham_line_algorithm


def run_algorithm(img, empty):
    """

    :param img:
    :param empty:
    :return:
        >>> img = 'd:/pictures/fydp/nudes4/2letters.jpg'
        >>> empty = 'd:/pictures/fydp/nudes4/black_empty.jpg'
        >>> img = 'd:/pictures/fydp/nudes3/3l_covered.jpg'
        >>> empty = 'd:/pictures/fydp/nudes3/empty_covered.jpg'
        >>> run_algorithm(img, empty)
    """
    img_ls = find_straight_lines(img)
    # empty_ls = find_straight_lines(empty)
    # empty_matrix = bresenham_line_algorithm(empty_ls)
    # matrix = bresenham_line_algorithm(img_ls) - empty_matrix
    # matrix[matrix < 0] = 0
    # all_paths, all_cycles = find_all_cycles(matrix)
    vertices, edges = convert_graph(img_ls)
    all_paths, all_cycles = find_cycles(vertices, edges)
    areas = pd.Series(np.unique(get_areas(all_paths, all_cycles)))
    # get_objects(areas)
    # object_areas = find_objects(areas)
    return get_objects(areas)

# def run_algorithm(img, empty):
#     """
#
#     :param img: Path to image for algorithm to run on
#     :param empty: An empty image as a base reference
#     :return: Dictionary of objects
#         >>> img = 'd:/pictures/fydp/nudes3/1b_covered.jpg'
#         >>> empty = 'd:/pictures/fydp/nudes3/empty_covered.jpg'
#         >>> run_algorithm(img, empty)
#     """
#     empty_edges = get_all_edges(empty)
#     img_edges = get_all_edges(img)
#     minned = matrix_min(img_edges.values()) - matrix_min(empty_edges.values())
#     # minned = matrix_min(img_edges.values())
#     minned[minned < 0] = 0
#
#     # cutoff pixels based on non-zero median
#     cutoff = 0
#     minned = ndimage.median_filter(minned, 4)
#     for x in range(3):
#         cutoff = np.nanmedian(np.nanmedian(np.where(minned != 0,
#                                                     minned,
#                                                     np.NaN)))
#         minned[minned < cutoff] = 0
#     minned[minned >= cutoff] = 1
#     all_paths, all_cycles = find_all_cycles(minned)
#     areas = pd.Series(get_areas(all_paths, all_cycles))
#     return get_objects(areas)
