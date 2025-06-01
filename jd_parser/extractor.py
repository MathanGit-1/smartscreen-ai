import pdfplumber
from docx import Document


def extract_text_from_pdf(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file_like_obj):
    return "\n".join([para.text for para in Document(file_like_obj).paragraphs])

def extract_text_from_txt(file_like_obj):
    return file_like_obj.read().decode("utf-8", errors="ignore")
