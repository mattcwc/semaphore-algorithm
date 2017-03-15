import numpy as np
import cv2
from image_algorithm.util.img_util import encode_vertex, max_x, max_y, expand_search_range


def find_straight_lines(img, plot=False):
    """

    :param img:
    :param min_length:
    :return:
    :usage:
        >>> img = 'd:/pictures/fydp/nudes3/3l_covered.jpg'
        >>> line_segments = find_straight_lines(img)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 20, 150)
    kwargs = {'rho': 0.01, 'theta': np.pi / 500, 'threshold': 3,
              'maxLineGap': 100}
    line_segments = cv2.HoughLinesP(edges, **kwargs)
    if plot:
        a, b, c = line_segments.shape
        for i in range(a):
            x1, y1, x2, y2 = line_segments[i][0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('lines', img)

    line_segments = [x[0].tolist() for x in line_segments]
    return line_segments


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
            matrix[b][a] = 1
        # matrix = bresenham_line(matrix, *ls)
        # x1, y1, x2, y2 = ls
        # dx, dy = x2 - x1, y2 - y1
        # abs_dx, abs_dy = abs(dx), abs(dy)  # absolute value
        # sx = 0 if dx == 0 else dx / abs_dx
        # sy = 0 if dy == 0 else dy / abs_dy
        # e = abs_dx - abs_dy
        # while x1 != x2 or y1 != y2:
        #     if (0 <= x1 < max_x) and (0 <= y1 < max_y):
        #         matrix[y1][x1] = 1
        #     e2 = e * 2
        #     if e2 > -abs_dy:
        #         e -= abs_dy
        #         x1 += sx
        #     if e2 < abs_dx:
        #         e += dx
        #         y1 += sy
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
        v1_idx = len(path) + 1
        adjacent = expand_search_range(edges.get(v1, None), sweep_range=50)
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
        >>> img = 'd:/pictures/fydp/nudes3/1b3l_covered.jpg'
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
