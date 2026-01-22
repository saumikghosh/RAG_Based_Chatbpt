from pypdf import PdfReader
def text_extracter(filepath):
    pdf_file = PdfReader(filepath)
    content = ''
    for page in pdf_file.pages:
        content = content + page.extract_text() + '\n'
    return content