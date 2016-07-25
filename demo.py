from util.firebase_util import put_delivery


def main():
    email = 'mailbox@semaphore.ca'
    secret = '56yZNdR1DApjhiKNKS3jcElWzWYSCWEfWPxjYHYf'
    database_url = 'https://smartbox-041.firebaseio.com'
    mailbox_id = 'temp_fd5c7ba5-c2db-4923-8075-046cbead8173'
    letters, magazines, newspapers, parcels = 0, 0, 0, 0
    put_delivery(database_url, email, secret, mailbox_id, letters, magazines, newspapers, parcels)


if __name__ == '__main__':
    main()
