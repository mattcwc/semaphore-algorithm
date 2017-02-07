from firebase import firebase
from math import ceil
from time import time

PUT_TYPES = ['deliveries', 'snapshots']


def put_data(db_url, email, secret, mailbox_id, put_type='deliveries',
             timestamp=None, letters=0, magazines=0, newspapers=0, parcels=0):
    """

    :param db_url:
    :param email:
    :param secret:
    :param mailbox_id:
    :param timestamp:
    :param letters:
    :param magazines:
    :param newspapers:
    :param parcels:
    :param put_type: How to put ('snapshots' or 'deliveries')
    :return:
    :usage:
        >>> from json import load
        >>> tmp = load(open('D:/Projects/fydp/semaphore-raspi/config'))
        >>> email = tmp.get('email')
        >>> secret = tmp.get('secret')
        >>> db_url = tmp.get('db_url')
        >>> mailbox_id = tmp.get('mailbox_id')
        >>> put_type = 'deliveries'
        >>> timestamp = None
        >>> letters, magazines, newspapers, parcels = 0, 0, 0, 1
        >>> put_data(db_url, email, secret, mailbox_id, put_type, timestamp
        >>>          letters, magazines, newspapers, parcels)
    """
    authentication = firebase.FirebaseAuthentication(secret, email)
    fb_app = firebase.FirebaseApplication(db_url, authentication)
    if timestamp is None:
        timestamp = int(ceil(time()))
    data = {'letters': letters, 'magazines': magazines,
            'newspapers': newspapers, 'parcels': parcels}
    assert put_type in PUT_TYPES, 'put_type param not in {}'.format(PUT_TYPES)
    return fb_app.put('/{}/{}'.format(put_type, mailbox_id), timestamp, data)


def get_snapshot(db_url, email, secret, mailbox_id):
    """

    :param db_url:
    :param email:
    :param secret:
    :param mailbox_id:
    :return:
    :usage:
        >>> from json import load
        >>> tmp = load(open('D:/Projects/fydp/semaphore-raspi/config'))
        >>> email = tmp.get('email')
        >>> secret = tmp.get('secret')
        >>> db_url = tmp.get('db_url')
        >>> mailbox_id = tmp.get('mailbox_id')
        >>> get_snapshot(db_url, email, secret, mailbox_id)
    """
    params = {'orderBy': '"$key"', 'limitToLast': 1}
    authentication = firebase.FirebaseAuthentication(secret, email)
    fb_app = firebase.FirebaseApplication(db_url, authentication)
    return fb_app.get('/snapshots/', mailbox_id, params=params)
