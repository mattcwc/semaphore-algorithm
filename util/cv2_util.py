"""
 Semaphore - Image Processing Algorithm
 Image Processing Algorithm component of Semaphore
 See https://shlchoi.github.io/semaphore/ for more information about Semaphore

 cv2_util.py
 Copyright (C) 2017 Matthew Chum, Samson H. Choi

 See https://github.com/mattcwc/semaphore-raspi/blob/master/LICENSE for license information
 """

import numpy as np
import cv2
from image_algorithm.util.img_util import encode_vertex, max_x, max_y, \
    expand_search_range


def read_image(img, convert_grey=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    if convert_grey:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def image_subtract(img, bg):
    return cv2.absdiff(img, bg)


def remove_background(img, e1, e2):
    diff1 = cv2.absdiff(img, e1)
    diff2 = cv2.absdiff(img, e2)
    return cv2.bitwise_and(diff1, diff2)


def find_straight_lines(img, plot=False):
    """

    :param img:
    :param plot:
    :return:
    :usage:
        >>> img = 'd:/google drive/fydp/images/img_sub test/1l_0.jpg'
        >>> img_ls = find_straight_lines(img, plot=True)
        >>> empty = 'd:/google drive/fydp/images/img_sub test/empty_0.jpg'
        >>> empty_ls = find_straight_lines(empty, plot=True)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 10, 150)
    kwargs = {'rho': 0.1, 'theta': np.pi / 512, 'threshold': 5,
              'maxLineGap': 50, 'minLineLength': 50}
    line_segments = cv2.HoughLinesP(edges, **kwargs)
    if plot:
        assert line_segments is not None, 'No line segements found'
        a, b, c = line_segments.shape
        for i in range(a):
            x1, y1, x2, y2 = line_segments[i][0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('img', img)

    line_segments = [x[0].tolist() for x in line_segments]
    return line_segments


def mask_shapes(img, plot=False):
    if isinstance(img, str):
        img = read_image(img)
    shape_mask = cv2.inRange(img, 0, np.mean(img))
    shape_mask = 255 - shape_mask
    if plot:
        cv2.namedWindow('mask', 1)
        cv2.imshow('mask', shape_mask)
    return shape_mask


def find_shapes(img):
    if isinstance(img, str):
        img = read_image(img)
    tmp = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
    contours = tmp[1]
    x, y = img.shape
    max_area = (x - 1) * (y - 1)  # Using this for cv2 moments area
    areas = []
    for c in contours:
        moment = cv2.moments(c)
        area = moment['m00']
        shape = detect_shape(c)
        print shape, area
        if area < max_area:
            areas.append(int(area))
        cv2.drawContours(img, [c], 0, (0, 0, 0), thickness=3)
    cv2.namedWindow('img', 1)
    cv2.imshow('img', img)
    return areas


def detect_shape(c):
    shape = None

    # Approximate contour and then the number of points
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
    # Square or Rectangle
    elif len(approx) == 4:
        shape = 'rectangle'
    elif len(approx) == 5:
        shape = 'pentagon'
    elif len(approx) == 6:
        shape = 'hexagon'
    else:
        shape = 'circle'

    # return the name of the shape
    return shape


def convert_graph(line_segments):
    """

    :param line_segments:
    :return: A directed graph
    :usage:
        >>> vertices, edges = convert_graph(line_segments)
    """
    vertices, edges = [], {}  # Returned variables
    vertex_check = {}
    for ls in line_segments:
        x1, y1, x2, y2 = ls
        v1, v2 = encode_vertex(x1, y1), encode_vertex(x2, y2)
        if vertex_check.get(v1, None) is None:
            vertex_check[v1] = 0
            vertices.append(v1)
            edges[v1] = []
        if vertex_check.get(v2, None) is None:
            vertex_check[v2] = 0
            vertices.append(v2)
            edges[v2] = []
        edges[v1] = v2
        edges[v2] = v1
    return vertices, edges


def bresenham_line(matrix, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:  # vertical line
        s = 0 if dy == 0 else dy / abs(dy)
        while y0 != y1:
            matrix[y0][x0] = 1
            y0 += s
    else:
        de = 0 if dx == 0 else abs(float(dy) / float(dx))
        e = de - 0.5
        for x in range(x0, x1 + 1):
            matrix[y0][x] = 1
            e += de
            if e >= 0.5:
                y0 += 1
                e -= 1
    return matrix


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def bresenham_line_algorithm(line_segments):
    """

    :param line_segments:
    :return:
    :usage:
        >>> img = 'd:/pictures/fydp/nudes3/1b3l_covered.jpg'
        >>> line_segments = find_straight_lines(img)
        >>> matrix = bresenham_line_algorithm(line_segments)
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(matrix, cmap=plt.cm.gray)
    """
    matrix = np.zeros((max_x, max_y), dtype=float)
    for ls in line_segments:
        x0, y0, x1, y1 = ls
        points = get_line((x0, y0), (x1, y1))
        for a, b in points:
            matrix[a][b] = 1
    return matrix


def cycle_traverse(start_vertex, edges):
    """
    Iterative DFS to find all cycles connected to the start vertex
    :param start_vertex: start vertex
    :param edges: uni-directional edges
    :return:
    """
    stack = [start_vertex]
    path, path_checker = [], {}  # currently traversing path
    cycle_map = {}  # Cycles in current path
    paths, cycles, prev_map = [], [], {}  # Returned variables
    while stack:  # Stack is not empty
        v1 = stack.pop()
        adjacent = expand_search_range(edges.get(v1, None), sweep_range=80)
        prev = path[-1] if len(path) > 0 else -1  # O(1)
        if prev > 0 and prev not in adjacent:
            # The previous path is finished and we continue to the next one
            # Save the paths and cycles if there are cycles, and refresh
            if cycle_map:
                paths.append(path)
                cycles.append(cycle_map)
            path = path[:path.index(prev) + 1]
            path_checker = {k: i for i, k in enumerate(path)}
            cycle_map = {}
        v1_idx = len(path)
        for v2 in adjacent:
            if v2 == prev:  # Ignore the vertex going back
                continue
            v2_idx = path_checker.get(v2, None)
            if v2_idx is None:  # If vertex isn't in the path yet
                if edges.get(v2, None) is not None:
                    # If v2 connects v1 to a vertex in the graph,
                    # add it to the DFS, otherwise it's not a good edge
                    prev_map[v2] = v1
                    if v2 not in stack:
                        stack.append(v2)
            else:  # If it's in the path then obviously there's a cycle
                if cycle_map.get(v2_idx, None) is None:
                    cycle_map[v2_idx] = []
                cycle_map[v2_idx].append(v1_idx)
        path.append(v1)
        path_checker[v1] = v1_idx
    return paths, cycles, prev_map


def find_cycles(vertices, edges):
    """

    :param vertices:
    :param edges:
    :return:
    :usage:
        >>> img = 'd:/google drive/fydp/images/img_sub test/1l_0.jpg'
        >>> line_segments = find_straight_lines(img)
        >>> vertices, edges = convert_graph(line_segments)
        >>> all_paths, all_cycles = find_cycles(vertices, edges)
    """
    all_paths, all_cycles = [], []
    ignore = {}
    for vertex in vertices:
        if ignore.get(vertex, None) is None:
            paths, cycles, prev_map = cycle_traverse(vertex, edges)
            all_paths.extend(paths)
            all_cycles.extend(cycles)
            ignore.update(prev_map)
    return all_paths, all_cycles
