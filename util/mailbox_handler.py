from json import load, dumps
from os.path import isfile
from subprocess import call
from util.firebase_util import put_delivery


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
        return put_delivery(self.db_url, self._email, self._secret,
                            self._mailbox_id,
                            letters, magazines, newspapers, parcels)

    def take_image(self, image_path):
        call(['raspistill', '-o', '{}.jpg'.format(image_path)])

    def process_image(self, image_path):
        pass
