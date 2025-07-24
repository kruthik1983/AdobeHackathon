import fitz 
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
        cannot be processed.
    """
    try:
        doc = fitz.open(file_path)
        all_pages_data = []
        for page_num, page in enumerate(doc):

            page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
            page_dict["page_number"] = page_num + 1
            page_dict["page_rect"] = page.rect
            all_pages_data.append(page_dict)
        doc.close()
        return all_pages_data
    except Exception as e:
        print(f"Error processing PDF file {os.path.basename(file_path)}: {e}")
        return None