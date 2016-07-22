from firebase import firebase
from math import ceil
from time import time



def put_delivery(database_url, email, secret, mailbox_id, letters, magazines, newspapers, parcels):
    """

    :param database_url:
    :param authentication:
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
        >>> letters, magazines, newspapers, parcels = 6, 6, 6, 6
        >>> put_delivery(database_url, email, secret, mailbox_id, letters, magazines, newspapers, parcels)
    """
    _authentication = firebase.FirebaseAuthentication(secret, email)
    _firebase = firebase.FirebaseApplication(database_url, _authentication)
    _timestamp =  int(ceil(time()))
    _data = {'letters': letters, 'magazines': magazines, 'newspapers': newspapers, 'parcels': parcels,
             'timestamp': _timestamp, 'mailbox': mailbox_id}
    _delivery_id = '{}_{}'.format(mailbox_id, _timestamp)
    return _firebase.put('/deliveries', _delivery_id, _data)
