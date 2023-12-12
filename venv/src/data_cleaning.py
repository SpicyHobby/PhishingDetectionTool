from bs4 import BeautifulSoup
import re

def clean_email_body(body):
    # Remove HTML tags
    soup = BeautifulSoup(body, "html.parser")
    cleaned_text = soup.get_text()

    # Lowercase the text
    cleaned_text = cleaned_text.lower()

    # Remove email specific characters and excessive whitespace
    cleaned_text = re.sub(r'\n|\r', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text
