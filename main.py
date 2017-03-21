from image_algorithm.util.img_util import get_areas, get_all_edges, \
    find_all_cycles
from image_algorithm.util.fuzzy_logic import get_objects, find_objects
import pandas as pd

from util.cv2_util import find_straight_lines, convert_graph, find_cycles, \
    read_image, remove_background, mask_shapes


def run_algorithm_old(img, e1, e2):
    """

    :param img:
    :param e1: First empty image, order of empties don't matter.
    :param e2: Second empty image
    :return:
        >>> e1 = 'd:/google drive/fydp/images/calibrate/calibrate_0.jpg'
        >>> e2 = 'd:/google drive/fydp/images/calibrate/calibrate_1.jpg'
        >>> img = 'd:/google drive/fydp/images/1box_1letter = 2 letters .jpg'
        >>> run_algorithm_old(img, e1, e2)
    """
    img = read_image(img, True)
    e1 = read_image(e1, True)
    e2 = read_image(e2, True)
    try:
        img = remove_background(img, e1, e2)
        img_ls = find_straight_lines(img)
        vertices, edges = convert_graph(img_ls)
        all_paths, all_cycles = find_cycles(vertices, edges)
        areas = pd.Series(get_areas(all_paths, all_cycles)).unique()
        # object_areas = find_objects(areas)
        object_areas = areas
    except AssertionError as e:
        # TODO: logger
        print str(e)
        object_areas = pd.Series([])
    return get_objects(object_areas)


def run_algorithm(img, empty):
    """

    :param img:
    :param empty: Placeholder variable so that server doesn't need to change
    :return:
    :usage:
        >>> img = 'd:/google drive/fydp/images/1l_r2l_2.jpg'
        >>> run_algorithm(img, None)
    """
    # Find objects
    img = read_image(img, True)
    contour = mask_shapes(img, True)
    tmp = contour.astype(int) - img.astype(int)
    tmp[tmp < 0] = 0
    tmp = tmp.astype('uint8')
    edges = get_all_edges(tmp)
    single = edges.get('canny')

    # Find cycles and areas
    all_paths, all_cycles = find_all_cycles(single)
    areas = pd.Series(get_areas(all_paths, all_cycles)).unique()
    object_areas = find_objects(areas)
    object_dict = get_objects(object_areas)

    # Pseudo-normalization of result
    div = float(sum(object_dict.values())) / 2  # only supports 2 items for now
    if div > 1:  # Don't want to increase the number of items
        object_dict = {k: int(round(v / div)) for k, v in
                       object_dict.iteritems()}

    # Check if it's only letters that have values. Empirical tests show that
    # mailboxes with only letters tend to produce double the amount.
    i = iter(object_dict.values())
    if any(i) and not any(i) and object_dict.get('letters') > 0:
        object_dict['letters'] /= 2
    return object_dict
