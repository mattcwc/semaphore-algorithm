from util.mailbox_handler import MailboxHandler


def main():
    _email = 'mailbox@semaphore.ca'
    _mailbox_id = 'temp_fd5c7ba5-c2db-4923-8075-046cbead8173'
    _mailbox = MailboxHandler(_email, _mailbox_id)
    _letters, _magazines, _newspapers, _parcels = 0, 0, 0, 0
    _mailbox.send_data(_letters, _magazines, _newspapers, _parcels)

if __name__ == '__main__':
    main()
