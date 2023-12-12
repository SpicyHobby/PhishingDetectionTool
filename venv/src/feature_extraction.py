import os
import email
from email.policy import default
from email.header import decode_header
from bs4 import BeautifulSoup
import re
import pandas as pd
from textblob import TextBlob

# Read an .eml file and return the email message object
def read_email(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        msg = email.message_from_file(file, policy=default)
    return msg

# Check sender's domain and analyze URL features
def check_sender_domain_and_url(msg):
    sender = msg.get('From', '')
    domain = sender.split('@')[-1] if sender else ''
    url_features = {
        'sender_domain': domain,
        'domain_length': len(domain),
        'special_chars_in_domain': sum(not c.isalnum() for c in domain),
        'is_secure': domain.startswith('https')
    }
    return url_features

# Extract hyperlinks from the email body
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

# Check for attachments
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
    suspicious_keywords_body = ['password', 'click', 'update', 'urgent', 'login', 'confirm', 'secure', ...]

    # Sentiment Analysis - Determines if the email text has a tone of urgency or threat
    sentiment = TextBlob(cleaned_text).sentiment.polarity

    features = {
        'contains_suspicious_keyword_body': any(keyword in cleaned_text for keyword in suspicious_keywords_body),
        'urgent_tone': sentiment < -0.5,  # Assuming negative sentiment indicates urgency or threat
    }
    return features


# Extract features from an email
def extract_features(email_path):
    features = {}
    msg = read_email(email_path)

    # URL and domain features
    features.update(check_sender_domain_and_url(msg))

    # Hyperlink analysis
    features['hyperlink_count'] = extract_hyperlinks(msg)

    # Subject and content analysis
    features['urgent_subject'] = subject_analysis(msg)
    features['has_attachment'] = attachment_analysis(msg)

    # Extract subject-based features
    subject_features = subject_analysis(msg)
    features.update(subject_features)

    # Extract text-based features from the email body
    cleaned_body = clean_email_body(msg.get_payload(decode=True))
    features.update(extract_text_features(cleaned_body))

    return features

# Process all .eml files in a directory and extract features
def process_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.eml'):
            file_path = os.path.join(directory, filename)
            features = extract_features(file_path)
            features['filename'] = filename  # Optionally track the file name
            data.append(features)
    return data