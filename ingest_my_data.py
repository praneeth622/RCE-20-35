import os
import re
from pypdf import PdfReader

DOCS_DIR = "./docs"
PDF_PATH = "RegsNavyIV.pdf"

def clean_filename(text):
    # Remove invalid characters for filenames
    return re.sub(r'[\\/*?:"<>|]', "", text).replace(" ", "_")

def ingest_pdf():
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    reader = PdfReader(PDF_PATH)
    count = 0
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        text = text.strip()
        if not text:
            continue
            
        # Create a short clean filename per page (representing a document here)
        filename = f"page_{i+1:03d}.txt"
        file_path = os.path.join(DOCS_DIR, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        count += 1
        
    print(f"Successfully extracted {count} documents (pages) to {DOCS_DIR}")

if __name__ == "__main__":
    ingest_pdf()
