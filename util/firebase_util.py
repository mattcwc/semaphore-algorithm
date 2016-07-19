from urllib2 import Request, urlopen
from json import dumps
from datetime import datetime

FIREBASE_URL = 'https://smartbox-041.firebaseio.com/raspi'


def post_json(post_data):
    if isinstance(post_data.get('date'), datetime):
        post_data['date'] = post_data.get('date').strftime('%Y-%m-%d %H:%M:%S')
    _req = Request(FIREBASE_URL)
    _req.add_header('Content-Type', 'application/json')
    _send_data = dumps(post_data)
    return urlopen(_req, _send_data)
