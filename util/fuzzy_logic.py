import pandas as pd


def get_membership_functions():
    x, y = 640, 480
    area = [float(a) for a in range(x * y)]
    l = pd.Series(0.0, index=area, name='letters')
    n = pd.Series(0.0, index=area, name='newspapers')
    m = pd.Series(0.0, index=area, name='magazines')
    p = pd.Series(0.0, index=area, name='parcels')
    l[500:2000] = (l[500:2000].index - 500)/2000
    l[2000:3000] = (3000 - l[2000:3000].index)/2000
    p[20000: 50000] = (p[20000: 50000].index - 20000)/50000
    p[50000: 100000] = (100000 - p[50000: 100000].index)/50000
    return pd.DataFrame([l, n, m, p]).transpose()


def find_objects(areas, threshold=30):
    """
    >>> object_areas = find_objects(areas)
    >>> get_objects(object_areas)
    :param areas:
    :param threshold:
    :return:
    """
    if not isinstance(areas, pd.Series):
        areas = pd.Series(areas)
    areas = areas.sort(inplace=False)
    p_area = None
    object_areas = []
    for a, area in areas.iteritems():
        if p_area is not None and abs(area - p_area) > threshold:
            object_areas.append(area)
        p_area = area
    return object_areas


def determine_object(area, mf=None):
    if mf is None:
        mf = get_membership_functions()
    slice = mf.iloc[area]
    if all(slice == 0):
        return None
    return slice.idxmax()


def get_objects(object_areas):
    mf = get_membership_functions()
    object_dict = {'letters': 0,
                   'newspapers': 0,
                   'magazines': 0,
                   'parcels': 0}
    for area in object_areas:
        item = determine_object(area, mf)
        if item is not None:
            object_dict[item] += 1
    return object_dict
