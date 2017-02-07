from util.firebase_util import put_data, get_snapshot


def process_data(db_url, email, secret, mailbox_id,
                 letters=0, magazines=0, newspapers=0, parcels=0):
    params = locals()
    prev_snap = get_snapshot(db_url, email, secret, mailbox_id)
    put_data(db_url, email, secret, mailbox_id, 'snapshot', None,
             letters, magazines, newspapers, parcels)

    if len(prev_snap) > 0:
        prev_snap = prev_snap.values()[0]
        for k, v in prev_snap.iteritems():
            if k in params:  # Effectively removes the 'timestamp' key
                params[k] = max(params[k] - v, 0)
    params['put_type'] = 'deliveries'
    put_data(**params)
