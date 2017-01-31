from skimage.io import imread
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny


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


def comparison_test(img):
    if isinstance(img, str):
        img = imread(img, as_grey=True)
    edges = {}
    for m in EDGE_METHODS.keys():
        edges[m] = get_edges(img, m)
    # TODO: output stuff
