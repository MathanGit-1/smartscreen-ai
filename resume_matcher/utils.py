import re
import io
import pandas as pd

def extract_mobile(text):
    # Match international numbers like +1-(234)-555-1234 or Indian 9876543210
    pattern = r'(\+?\d{1,3}[-\s]?)?(\(?\d{3}\)?[-\s]?)\d{3}[-\s]?\d{4}|\b[6-9]\d{9}\b'
    match = re.search(pattern, text)

    if match:
        number = match.group().strip()

        # Mask India 10-digit numbers only
        if re.fullmatch(r'[6-9]\d{9}', number):
            return re.sub(r'(\d{2})\d{6}(\d{2})', r'\1XXXXXX\2', number)
        return number  # Return international number as-is
    return "Not found"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else "Not found"

def clean_skills(raw_skills):
    return sorted(set(s.strip().title() for s in raw_skills if s.strip()))

