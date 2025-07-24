import re
import fitz 

def is_bold(font_name: str) -> bool:
    """Checks if a font name suggests a bold weight."""
    return any(val in font_name.lower() for val in ['bold', 'black', 'heavy'])

def starts_with_numbering(text: str) -> bool:
    """
    Checks if a string starts with a common numbering pattern (e.g., "1.", "1.2", "A.").
    This is a strong, language-agnostic indicator of a heading.[4]
    """
    pattern = re.compile(r"^\s*(\d+(\.\d+)*|[A-Z]\.|[a-z]\.)\s+")
    return bool(pattern.match(text))

def extract_features(pages_data):
    """
    Processes raw page data to extract a feature vector for each text line.

    This function iterates through the structured data from the PDF parser,
    aggregates text lines, and computes a set of features for each line
    that can be used to classify it as a heading or body text.

    Args:
        pages_data: The list of page dictionaries from pdf_parser.

    Returns:
        A list of dictionaries, where each dictionary is a feature vector
        for a text line.
    """
    all_lines_features = []
    prev_bbox_y1 = 0.0

    for page in pages_data:
        page_width = page["page_rect"].width
        
        blocks = sorted(page.get("blocks",), key=lambda b: b.get("bbox", (0,0,0,0))[1])

        for block in blocks:
            print(block)
            if block.get("type") == 0: 
                for line in block["lines"]:
                    if not line["spans"]:
                        continue

                    line_text = "".join([span["text"] for span in line["spans"]]).strip()
                    if not line_text:
                        continue

                    first_span = line["spans"][0]
                    font_size = first_span["size"]
                    font_name = first_span["font"]
                    line_bbox = fitz.Rect(line["bbox"])


                    feature_vector = {
                        "text": line_text,
                        "page_number": page["page_number"],
                        "bbox": line_bbox,
                        "font_size": font_size,
                        "font_name": font_name,
                        "is_bold": is_bold(font_name),
                        "is_italic": "italic" in font_name.lower() or "oblique" in font_name.lower(),
                        "is_all_caps": line_text.isupper() and any(c.isalpha() for c in line_text),
                        "starts_with_numbering": starts_with_numbering(line_text),
                        "text_length": len(line_text),
                        "word_count": len(line_text.split()),
                        "is_centered": abs(page_width / 2 - (line_bbox.x0 + line_bbox.x1) / 2) < (page_width * 0.1),
                        "space_above": line_bbox.y0 - prev_bbox_y1,
                    }
                    all_lines_features.append(feature_vector)
                    prev_bbox_y1 = line_bbox.y1
    
    return all_lines_features