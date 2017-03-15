import pandas as pd


def get_membership_functions():
    x, y = 640, 480
    area = [float(a) for a in range(x * y)]
    l = pd.Series(0.0, index=area, name='letters')
    n = pd.Series(0.0, index=area, name='newspapers')
    m = pd.Series(0.0, index=area, name='magazines')
    p = pd.Series(0.0, index=area, name='parcels')
    l[1000:3000] = (l[1000:3000].index - 1000) / 3000
    l[3000:5000] = (5000 - l[3000:5000].index) / 3000
    p[6000: 8000] = (p[6000: 8000].index - 6000) / 8000
    p[8000: 10000] = (10000 - p[8000: 10000].index) / 8000
    return pd.DataFrame([l, n, m, p]).transpose()


def find_objects(areas, threshold=30):
    """
    May potentially work, but needs re-working
    >>> object_areas = find_objects(areas)
    >>> get_objects(object_areas)
    :param areas:
    :param threshold:
    :return:
    """
    if not isinstance(areas, pd.Series):
        areas = pd.Series(areas)
    areas = areas.sort_values(inplace=False)
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
    return slice.idxmax() if any(slice > 0) else None


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
    for k in object_dict.keys():
        object_dict[k] = int(round(float(object_dict[k]) / 40))
        while object_dict[k] > 20:
            object_dict[k] = int(round(float(object_dict[k]) / 20))
    object_dict['letters'] /= 2
    if object_dict['parcels'] > 2:
        object_dict['parcels'] = 0
    return object_dict
