import firebase_util
from datetime import datetime


def send_data(letter, parcel, magazine, newspaper):
    _data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'mail': {'letter': letter, 'parcel': parcel, 'magazine': magazine, 'newspaper': newspaper}}
    firebase_util.post_json(_data)