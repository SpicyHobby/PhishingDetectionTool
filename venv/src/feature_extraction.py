import os
import email
from email.policy import default
from email.header import decode_header
from bs4 import BeautifulSoup
import re
import pandas as pd
from textblob import TextBlob

# Reads an .eml file and returns the email message object
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

# Checks sender's domain and analyzes URL features
def check_sender_domain_and_url(msg):
    try:
        sender = msg.get('From', '')
        domain = sender.split('@')[-1] if sender else ''
    except AttributeError:
        # If parsing fails, sets default values
        sender = ''
        domain = ''

    url_features = {
        'sender_domain': domain,
        'domain_length': len(domain),
        'special_chars_in_domain': sum(not c.isalnum() for c in domain),
        'is_secure': domain.startswith('https')
    }
    return url_features

# Extracts hyperlinks from the email body
def extract_hyperlinks(msg):
    links = []
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            soup = BeautifulSoup(part.get_payload(decode=True), 'html.parser')
            links.extend([a['href'] for a in soup.find_all('a', href=True)])
    return len(links)

#subject_analysis function
def subject_analysis(msg):
    subject = msg.get('Subject', '')
    subject, encoding = decode_header(subject)[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or 'utf-8').lower()

    suspicious_keywords = ['urgent', 'important', 'response needed', 'verify', 'account', 'security', 'invoice',
                           'new', 'required', 'action', 'file', 'verification', 'document', 'efax', 'vm', 'message']
    features = {
        'blank_subject': subject.strip() == '',
        'contains_suspicious_keyword': any(re.search(r'\b' + keyword + r'\b', subject) for keyword in suspicious_keywords)
    }
    return features

# Checks for attachments
def attachment_analysis(msg):
    has_attachment = False
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart' and part.get('Content-Disposition') is not None:
            has_attachment = True
            break
    return has_attachment

#extract_text_features function
def extract_text_features(cleaned_text):
    #list of suspicious keywords
    suspicious_keywords_body = ['password', 'click', 'update', 'urgent', 'login', 'confirm', 'secure',
                                'account', 'credit', 'offer', 'free', 'winner']

    # Sentiment Analysis - Determines if the email text has a tone of urgency or threat
    sentiment = TextBlob(cleaned_text).sentiment.polarity

    features = {
        'contains_suspicious_keyword_body': any(keyword in cleaned_text for keyword in suspicious_keywords_body),
        'urgent_tone': sentiment < -0.5,  # Assuming negative sentiment indicates urgency or threat
    }
    return features


# Extracts features from an email
def extract_features(input_data, clean_content):
    # Define email_msg at the beginning of the function
    email_msg = None

    # Check if input_data is a file path or an EmailMessage object
    if isinstance(input_data, str):
        # It's a file path, read the email
        email_msg = read_email(input_data)
    elif isinstance(input_data, email.message.Message):
        # It's already an EmailMessage object
        email_msg = input_data
    else:
        raise ValueError("Invalid input type for extract_features")

    # Initialize all features with default values
    features = {
        'sender_domain': '',  # or any default value if needed
        'domain_length': 0,
        'special_chars_in_domain': 0,
        'is_secure': 0,  # False as default, will be converted to 0 later
        'hyperlink_count': 0,
        'blank_subject': 1,  # True (1) as default, change if needed
        'contains_suspicious_keyword': 0,  # False as default
        'has_attachment': 0,  # False as default
        'contains_suspicious_keyword_body': 0,  # False as default
        'urgent_tone': 0  # False as default
    }

    # Update features with actual values from email_msg
    features.update(check_sender_domain_and_url(email_msg))
    features['hyperlink_count'] = extract_hyperlinks(email_msg)
    features.update(subject_analysis(email_msg))
    features['has_attachment'] = attachment_analysis(email_msg)
    features.update(extract_text_features(clean_content))

    return features

# Proceses all .eml files in a directory and extracts features
def process_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.eml'):
            file_path = os.path.join(directory, filename)
            features = extract_features(file_path)
            features['filename'] = filename  # tracks the file name
            data.append(features)
    return data