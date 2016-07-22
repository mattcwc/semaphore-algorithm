import firebase_util as fu

class DataSender:
    def __init__(self, email=None, mailbox_id=None):
        self.db_url = 'https://smartbox-041.firebaseio.com'
        self.secret = '56yZNdR1DApjhiKNKS3jcElWzWYSCWEfWPxjYHYf'  # TODO: find a way to get this in a safer manner
        self.email = email
        self.mailbox_id = mailbox_id

    def save_mailbox(self, save_path):
        if self.email is None:
            raise ValueError('Email cannot be NoneType')
        elif self.mailbox_id is None:
            raise ValueError('Mailbox ID cannot be NoneType')
        else:
            # TODO: Haven't decided on how to do this
            pass


    def load_mailbox(self, load_path):
        # TODO: Save, then load
        pass

    def send_data(self, letters, magazines, newspapers, parcels):
        return fu.put_delivery(self.db_url, self.email, self.secret, self.mailbox_id,
                               letters, magazines, newspapers, parcels)