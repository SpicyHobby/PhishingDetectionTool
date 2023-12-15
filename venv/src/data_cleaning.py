from bs4 import BeautifulSoup
import re
import os

def clean_email_body(body):
    # Check if 'body' is a string and does not resemble a file path
    if isinstance(body, str) and not os.path.exists(body):
        soup = BeautifulSoup(body, "html.parser")
        cleaned_text = soup.get_text()
    else:
        # Handle unexpected 'body' content
        # For now, let's just return an empty string if 'body' is not as expected
        return ""

    # Lowercase the text
    cleaned_text = cleaned_text.lower()

    # Remove email specific characters and excessive whitespace
    cleaned_text = re.sub(r'\n|\r', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text
