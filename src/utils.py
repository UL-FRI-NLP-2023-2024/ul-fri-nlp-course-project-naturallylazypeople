import re


def clean_text(text):
    # Convert to lowercase
    cleaned_text = text.lower()

    # Remove special characters except whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # Remove extra whitespaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text
