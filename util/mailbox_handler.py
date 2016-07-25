from json import load, dumps


class MailboxHandler:
    def __init__(self, email=None, mailbox_id=None):
        self.db_url = 'https://smartbox-041.firebaseio.com'
        self.secret = '56yZNdR1DApjhiKNKS3jcElWzWYSCWEfWPxjYHYf'  # TODO: find a way to get this in a safer manner
        self.default_save_load_path = '/home/pi/'
        self.email = email
        self.mailbox_id = mailbox_id

    def save_mailbox(self, save_path):
        _str = dumps({'email': self.email, 'mailbox_id': self.mailbox_id})
        with open(save_path, 'w') as _txt_file:
            _txt_file.write(_str)
            _txt_file.close()

    def load_mailbox(self, load_path):
        with open(load_path, 'r') as _txt_file:
            _str = _txt_file.read()
            _data = load(_str)
            self.email = _data['email']
            self.mailbox_id = _data['mailbox_id']
            _txt_file.close()

    def send_data(self, letters, magazines, newspapers, parcels):
        from util.firebase_util import put_delivery
        return put_delivery(self.db_url, self.email, self.secret, self.mailbox_id,
                            letters, magazines, newspapers, parcels)

    def take_image(self, image_path):
        from subprocess import call
        call(['raspistill', '-o', '{}.jpg'.format(image_path)])

    def process_image(self, image_path):
        pass
