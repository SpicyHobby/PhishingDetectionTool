import email
from email.policy import default
from email.header import decode_header


def read_email(file_path):
    # List of encodings to try
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                msg = email.message_from_file(file, policy=default)
                return msg
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(f"Failed to decode {file_path} with tried encodings.")

def get_email_components(msg):
    # Extracts subject
    subject = msg.get('Subject', '')
    subject, encoding = decode_header(subject)[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or 'utf-8')

    # Extracts sender
    sender = msg.get('From', '')
    sender, encoding = decode_header(sender)[0]
    if isinstance(sender, bytes):
        sender = sender.decode(encoding or 'utf-8')

    # Extracts body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                body = part.get_payload(decode=True).decode('utf-8')
                break
    else:
        body = msg.get_payload(decode=True).decode('utf-8')

    return subject, sender, body
