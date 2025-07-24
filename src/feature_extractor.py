# src/feature_extractor.py
import re
import fitz # PyMuPDF

def is_bold(font_name: str) -> bool:
    """Checks if a font name suggests a bold weight, robust for multilingual fonts."""
    # Common indicators for bold/heavy fonts, works across many scripts
    return any(val in font_name.lower() for val in ['bold', 'black', 'heavy', 'demi', 'semibold', 'extrabold'])

def is_italic(font_name: str) -> bool:
    """Checks if a font name suggests an italic or oblique style."""
    return any(val in font_name.lower() for val in ['italic', 'oblique'])

def starts_with_numbering_or_bullet(text: str) -> bool:
    """
    Checks if a string starts with common numbering (1., 1.1, A., a.) or bullet patterns.
    Extended to include full-width numbers common in CJK languages and basic bullet points.
    """
    # Pattern for various numbering schemes:
    # 1. / 1.1.2 / (1) / [1]
    # A. / B)
    # a. / b)
    # Full-width numbers: １. / １．１
    # Common bullet characters: •, *, -, –, —
    pattern = re.compile(
        r"^\s*("
        r"(\d+\.)+\d*|"      # 1., 1.1, 1.1.1
        r"\(\d+\)|\[\d+\]|"   # (1), [1]
        r"[A-Z]\.?|[a-z]\.?|" # A., a.
        r"[A-Z]\)|[a-z]\)|"   # A), a)
        r"[０-９]+(\．[０-９]+)*|" # Full-width numbers for CJK
        r"[\u2022\u002A\u002D\u2013\u2014]" # Common bullet characters: •, *, -, –, —
        r")\s*", re.UNICODE
    )
    return bool(pattern.match(text))

def is_centered(line_bbox: fitz.Rect, page_width: float, tolerance_ratio: float = 0.05) -> bool:
    """
    Checks if a line is approximately centered on the page.
    Tolerance ratio determines how far from the true center it can be.
    """
    line_center_x = (line_bbox.x0 + line_bbox.x1) / 2
    page_center_x = page_width / 2
    return abs(page_center_x - line_center_x) < (page_width * tolerance_ratio)

def extract_features(pages_data):
    """
    Processes raw page data to extract a rich feature vector for each text line.
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

    for page_idx, page in enumerate(pages_data):
        page_number = page["page_number"]
        page_width = page["page_rect"].width
        page_height = page["page_rect"].height
        
        # Initialize prev_bbox_y1 for each new page to correctly calculate space_above
        prev_bbox_y1 = 0.0 

        # Sort blocks primarily by y0, then x0 for correct reading order
        blocks = sorted(page.get("blocks", []), key=lambda b: (b.get("bbox", (0,0,0,0))[1], b.get("bbox", (0,0,0,0))[0]))

        for block in blocks:
            # Only process text blocks (type 0)
            if block.get("type") == 0: 
                # Sort lines within a block by y0
                lines = sorted(block.get("lines", []), key=lambda l: l.get("bbox", (0,0,0,0))[1])
                
                for line in lines:
                    if not line.get("spans"):
                        continue

                    # Concatenate text from all spans in the line
                    line_text = "".join([span["text"] for span in line["spans"]]).strip()
                    if not line_text: # Skip empty lines
                        continue

                    # Aggregate font information from all spans in the line
                    # Using the dominant font size and checking for any bold/italic spans
                    dominant_font_size = 0
                    font_sizes = [s["size"] for s in line["spans"]]
                    if font_sizes:
                        dominant_font_size = max(set(font_sizes), key=font_sizes.count) # Most common font size

                    is_line_bold = any(is_bold(span["font"]) for span in line["spans"])
                    is_line_italic = any(is_italic(span["font"]) for span in line["spans"])
                    
                    # Use the font name of the first span for general font family identification
                    first_span_font_name = line["spans"][0]["font"]

                    line_bbox = fitz.Rect(line["bbox"])

                    space_above = line_bbox.y0 - prev_bbox_y1 if prev_bbox_y1 > 0 else 0
                    
                    feature_vector = {
                        "text": line_text,
                        "page_number": page_number,
                        "bbox": line_bbox,
                        "font_size": dominant_font_size,
                        "font_name": first_span_font_name, # Representative font name
                        "is_bold": is_line_bold,
                        "is_italic": is_line_italic,
                        "is_all_caps": line_text.isupper() and any(c.isalpha() for c in line_text),
                        "starts_with_pattern": starts_with_numbering_or_bullet(line_text),
                        "text_length": len(line_text),
                        "word_count": len(line_text.split()) if line_text.strip() else 0, # Handle empty string for split()
                        "is_centered": is_centered(line_bbox, page_width),
                        "space_above": space_above,
                        "x0": line_bbox.x0, # X-coordinate for indentation analysis
                        "y0": line_bbox.y0, # Y-coordinate for sorting
                        "page_height": page_height # For relative position calculations if needed
                    }
                    all_lines_features.append(feature_vector)
                    prev_bbox_y1 = line_bbox.y1
    
    return all_lines_features