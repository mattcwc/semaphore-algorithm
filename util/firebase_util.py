from firebase import firebase
from math import ceil
from time import time


def put_delivery(database_url, email, secret, mailbox_id, letters, magazines,
                 newspapers, parcels):
    """

    :param database_url:
    :param email:
    :param secret:
    :param mailbox_id:
    :param letters:
    :param magazines:
    :param newspapers:
    :param parcels:
    :return:
    :usage:
        >>> email = 'mailbox@semaphore.ca'
        >>> secret = '56yZNdR1DApjhiKNKS3jcElWzWYSCWEfWPxjYHYf'
        >>> database_url = 'https://smartbox-041.firebaseio.com'
        >>> mailbox_id = 'temp_fd5c7ba5-c2db-4923-8075-046cbead8173'
        >>> letters, magazines, newspapers, parcels = 0, 0, 0, 1
        >>> put_delivery(database_url, email, secret, mailbox_id, letters,
        >>>              magazines, newspapers, parcels)
    """
    _authentication = firebase.FirebaseAuthentication(secret, email)
    _firebase = firebase.FirebaseApplication(database_url, _authentication)
    _timestamp = int(ceil(time()))
    _data = {'letters': letters, 'magazines': magazines,
             'newspapers': newspapers, 'parcels': parcels}
    return _firebase.put('/deliveries/{}'.format(mailbox_id), _timestamp, _data)
