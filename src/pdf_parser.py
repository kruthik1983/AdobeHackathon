import fitz # PyMuPDF
import os

def process_pdf(file_path: str):
    """
    Opens a PDF and extracts structured text data from each page.
    This function uses PyMuPDF to parse a PDF and extract its content
    as a structured dictionary, which includes blocks, lines, and spans
    with detailed font and position information.
    Args:
        file_path: The path to the PDF file.
    Returns:
        A list of page dictionaries, where each dictionary contains the
        structured text data for a page. Returns None if the file
        cannot be processed or if it exceeds 50 pages.
    """
    try:
        doc = fitz.open(file_path)
        # Hackathon constraint check: maximum 50 pages
        if doc.page_count > 50:
            print(f"Warning: PDF {os.path.basename(file_path)} has {doc.page_count} pages, exceeding the 50-page limit. Only processing the first 50 pages.")
        
        all_pages_data = []
        for page_num, page in enumerate(doc):
            if page_num >= 50: # Enforce max 50 pages strictly
                break
            page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
            page_dict["page_number"] = page_num + 1
            page_dict["page_rect"] = page.rect # Store page dimensions for relative positioning
            all_pages_data.append(page_dict)
        doc.close()
        return all_pages_data
    except Exception as e:
        print(f"Error processing PDF file {os.path.basename(file_path)}: {e}")
        return None