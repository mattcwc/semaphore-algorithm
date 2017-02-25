from json import load, dumps
from os.path import isfile
from subprocess import call
import numpy as np
from util.firebase_util import put_data
from util.img_util import get_all_edges, convert_adjacency_list, decode_vertex
from util.math_func import matrix_min, cycle_finder, area_in_cycle


class MailboxHandler:
    def __init__(self, email=None, mailbox_id=None, config_path='config'):
        self.db_url = 'https://smartbox-041.firebaseio.com'
        self._secret = ''
        self.default_path = '/home/pi/'
        self._email = email
        self._mailbox_id = mailbox_id
        if isfile(config_path):
            self.load_mailbox(config_path)

    def save_mailbox(self, save_path):
        _str = dumps({'secret': self._secret, 'email': self._email,
                      'mailbox_id': self._mailbox_id})
        with open(save_path, 'w') as _txt_file:
            _txt_file.write(_str)
            _txt_file.close()

    def load_mailbox(self, config_path):
        with open(config_path, 'r') as _txt_file:
            _data = load(_txt_file)
            self._secret = _data['secret']
            self._email = _data['email']
            self._mailbox_id = _data['mailbox_id']
            _txt_file.close()

    def send_data(self, letters, magazines, newspapers, parcels):
        return put_data(self.db_url, self._email, self._secret,
                        self._mailbox_id, 'deliveries', None,
                        letters, magazines, newspapers, parcels)

    @staticmethod
    def take_image(img_path):
        call(['raspistill', '-o', '{}.jpg'.format(img_path)])

    @staticmethod
    def process_image(img):
        edges = get_all_edges(img)
        minned = matrix_min(edges.values())

        # cutoff pixels based on non-zero median
        cutoff = np.nanmedian(
            np.nanmedian(np.where(minned != 0, minned, np.NaN)))
        minned[minned < cutoff] = 0
        graph = convert_adjacency_list(minned)  # Also encodes vertices
        cycles = cycle_finder(*graph)

        # Decoding vertices and getting areas
        areas = []
        for c in cycles:
            cycle = [decode_vertex(v) for v in c]
            areas.append(area_in_cycle(cycle))
