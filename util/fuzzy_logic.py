"""
 Semaphore - Image Processing Algorithm
 Image Processing Algorithm component of Semaphore
 See https://shlchoi.github.io/semaphore/ for more information about Semaphore

 fuzzy_logic.py
 Copyright (C) 2017 Matthew Chum

 See https://github.com/mattcwc/semaphore-raspi/blob/master/LICENSE for license information
 """

import numpy as np
import pandas as pd


def get_membership_functions():
    x, y = 640, 480
    area = [float(a) for a in range(x * y)]
    l = pd.Series(0.0, index=area, name='letters')
    n = pd.Series(0.0, index=area, name='newspapers')
    m = pd.Series(0.0, index=area, name='magazines')
    p = pd.Series(0.0, index=area, name='parcels')
    l[800:7000] = (l[800:7000].index - 800) / 7000
    l[7000:12000] = (12000 - l[7000:12000].index) / 7000
    p[45000: 140000] = (p[45000: 140000].index - 45000) / 140000
    p[140000: 200000] = (200000 - p[140000: 200000].index) / 140000
    return pd.DataFrame([l, n, m, p]).transpose()


def find_objects(areas, threshold=30):
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
        if area >= len(mf):
            continue
        item = determine_object(area, mf)
        if item is not None:
            object_dict[item] += 1

    # Pseudo-normalization of result
    div = np.mean(object_dict.values())
    if div > 1:  # Don't want to increase the number of items
        object_dict = {k: int(np.floor(v / div)) for k, v in
                       object_dict.iteritems()}

    # Check if it's only letters that have values. Empirical tests show that
    # mailboxes with only letters tend to produce more. Quick hack to try and
    # make it a little closer
    i = iter(object_dict.values())
    if any(i) and not any(i) and object_dict.get('letters') > 0:
        object_dict['letters'] /= 2
    return object_dict
