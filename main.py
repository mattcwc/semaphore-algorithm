from image_algorithm.util.img_util import get_areas, get_all_edges, \
    find_all_cycles
from image_algorithm.util.fuzzy_logic import get_objects, find_objects
import numpy as np
import pandas as pd

from image_algorithm.util.cv2_util import find_straight_lines, convert_graph, find_cycles, \
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


def run_algorithm(img, empty, plot=False):
    """

    :param img:
    :param empty: Placeholder variable so that server doesn't need to change
    :return:
    :usage:
        >>> img = 'd:/google drive/fydp/images/3l_r1l1b.jpg'
        >>> run_algorithm(img, None)
    """
    # Find objects
    img = read_image(img, True)
    contour = mask_shapes(img)
    tmp = contour.astype(int) - img.astype(int)
    tmp[tmp < 0] = 0
    tmp = tmp.astype('uint8')

    if plot:
        import cv2
        cv2.namedWindow('tmp', 1)
        cv2.imshow('tmp', tmp)

    edges = get_all_edges(tmp)
    single = edges.get('canny')

    # Find cycles and areas
    all_paths, all_cycles = find_all_cycles(single)
    areas = pd.Series(get_areas(all_paths, all_cycles)).unique()
    object_areas = find_objects(areas)
    object_dict = get_objects(object_areas)

    return object_dict


def algorithm_test_multiple(images):
    """

    :param images:
    :return:

    """
    ret = {}
    for img in images:
        print 'image: {}'.format(img)
        object_dict = run_algorithm(img, None)
        ret[img] = object_dict
        print 'objects: {}'.format(object_dict)
    return ret


def algorithm_test(folder_path):
    """

    :param folder_path:
    :return:
    :usage:
        >>> folder_path = 'd:/google drive/fydp/images/letters_only/'
        >>> algorithm_test(folder_path)
    """
    from os import listdir, path
    images = []
    for f in listdir(folder_path):
        if f.endswith('.jpg'):
            img = path.join(folder_path, f)
            images.append(img)
    return algorithm_test_multiple(images)
