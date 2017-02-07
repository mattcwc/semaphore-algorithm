from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny
from matplotlib import pyplot as plt


EDGE_METHODS = {'roberts': roberts,
                'sobel': sobel,
                'scharr': scharr,
                'prewitt': prewitt,
                'canny': canny}


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
    if isinstance(img, str):
        img = imread(img, as_grey=True)
    edges = {}
    for i, m in enumerate(EDGE_METHODS.keys()):
        edges[m] = get_edges(img, m)
    if plot:
        fig, ax = plt.subplots(1, len(EDGE_METHODS))
        for i, (m, e) in enumerate(edges.iteritems()):
            ax[i].set_title(m)
            ax[i].imshow(e, cmap=plt.cm.gray)

    return edges


def combine_edges(edges_list, how='maxmin'):
    """

    :param edges_list: All 2d edge arrays to be aggregated as list
    :param how: Method of aggregation
    :return: aggregated 2d array of edges
    """
    pass
