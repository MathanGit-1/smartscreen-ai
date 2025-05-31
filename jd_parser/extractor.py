import fitz  # PyMuPDF
from docx import Document


def extract_text_from_pdf(file_like_obj):
    try:
        print("📄 Opening PDF stream...")
        doc = fitz.open("pdf", file_like_obj.read())
        print(f"📄 PDF has {len(doc)} pages.")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"❌ PDF Error: {str(e)}")
        return f"❌ Failed to read file: {str(e)}"

def extract_text_from_docx(file_like_obj):
    return "\n".join([para.text for para in Document(file_like_obj).paragraphs])

def extract_text_from_txt(file_like_obj):
    return file_like_obj.read().decode("utf-8", errors="ignore")
