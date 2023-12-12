import email
from email.policy import default
from email.header import decode_header

def read_email(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        msg = email.message_from_file(file, policy=default)
    return msg

def get_email_components(msg):
    # Extract subject
    subject = msg.get('Subject', '')
    subject, encoding = decode_header(subject)[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or 'utf-8')

    # Extract sender
    sender = msg.get('From', '')
    sender, encoding = decode_header(sender)[0]
    if isinstance(sender, bytes):
        sender = sender.decode(encoding or 'utf-8')

    # Extract body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                body = part.get_payload(decode=True).decode('utf-8')
                break
    else:
        body = msg.get_payload(decode=True).decode('utf-8')

    return subject, sender, body
