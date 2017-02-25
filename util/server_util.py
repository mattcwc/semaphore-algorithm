from json import load
from os.path import isfile
from flask import Flask, request, abort
from handlers.client_handler import process_data
from time import time

app = Flask(__name__)


def load_config(config_path):
    if isfile(config_path):
        config = load(open(config_path, 'r'))
        return config
    return None


@app.route("/", methods=['POST'])
def snapshot():
    if u'mailbox' not in request.json.keys():
        abort(400, 'Mailbox ID was not provided')

    mailbox = request.json.get(u'mailbox')

    letters = int(request.json.get(u'letters', 0))
    magazines = int(request.json.get(u'magazines', 0))
    newspapers = int(request.json.get(u'newspapers', 0))
    parcels = int(request.json.get(u'parcels', 0))

    process_data(app.config['db_url'], app.config['email'],
                 app.config['secret'], mailbox,
                 letters, magazines, newspapers, parcels)
    return '', 200


def time_run(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        ret = func(*args, **kwargs)
        run_time = time() - t1
        func_name = func.__name__
        print 'function <{}> took {} seconds to run'.format(func_name, run_time)
        return ret

    return wrapper
