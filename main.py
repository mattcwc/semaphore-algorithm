from util.img_util import get_all_edges, find_all_cycles, get_areas, imread
from util.math_func import matrix_min
from util.fuzzy_logic import get_objects
from scipy import ndimage
import numpy as np
import pandas as pd


def run_algorithm(img, empty):
    empty_edges = get_all_edges(empty)
    #img_edges = get_all_edges(imread(img, as_grey=True) - imread(empty, as_grey=True))
    # minned = matrix_min(img_edges.values()) - matrix_min(empty_edges.values())
    img_edges = get_all_edges(img)
    minned = matrix_min(img_edges.values()) - matrix_min(empty_edges.values())
    # minned = matrix_min(img_edges.values())
    minned[minned < 0] = 0

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
    return get_objects(areas)
#
#
# if __name__ == '__main__':
#     run_algorithm('/home/derp/pic.jpg', '/home/derp/empty.jpg')
