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
        return Noneimport fitz  # PyMuPDF
import os

def process_pdf(file_path: str):
    """
    Opens a PDF and extracts structured text data, intelligently merging text lines
    and excluding content from tables from the entire document.
    """
    try:
        doc = fitz.open(file_path)
        print(f"Processing '{os.path.basename(file_path)}' with {doc.page_count} pages.")
        
        all_pages_data = []
        # The 50-page limit has been removed to process the entire document.
        for page_num, page in enumerate(doc):

            # 1. Find table areas to exclude them from text extraction
            table_bboxes = [fitz.Rect(table.bbox) for table in page.find_tables()]

            # 2. Get the raw text dictionary from the page
            page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
            
            # 3. Filter out any text blocks that fall within a detected table
            filtered_blocks = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Process only text blocks
                    block_bbox = fitz.Rect(block["bbox"])
                    is_in_table = any(block_bbox.intersects(table_bbox) for table_bbox in table_bboxes)
                    if not is_in_table:
                        filtered_blocks.append(block)
            
            # 4. Perform line merging within each remaining block
            for block in filtered_blocks:
                if "lines" not in block or not block["lines"]:
                    continue

                merged_lines = []
                if not block["lines"]:
                    continue
                current_line = block["lines"][0]

                for i in range(1, len(block["lines"])):
                    next_line = block["lines"][i]
                    
                    # Check if the next line is on the same vertical level (y0)
                    if abs(current_line["bbox"][1] - next_line["bbox"][1]) < 2: # 2-point tolerance
                        # Merge spans and update bounding box
                        current_line["spans"].extend(next_line["spans"])
                        current_line["bbox"] = (
                            min(current_line["bbox"][0], next_line["bbox"][0]),
                            min(current_line["bbox"][1], next_line["bbox"][1]),
                            max(current_line["bbox"][2], next_line["bbox"][2]),
                            max(current_line["bbox"][3], next_line["bbox"][3]),
                        )
                        # Sort spans by horizontal position (x0) after merging
                        current_line["spans"].sort(key=lambda s: s["bbox"][0])
                    else:
                        merged_lines.append(current_line)
                        current_line = next_line
                
                merged_lines.append(current_line)
                block["lines"] = merged_lines

            # 5. Re-assemble the page dictionary with the processed blocks
            page_dict["blocks"] = filtered_blocks
            page_dict["page_number"] = page_num + 1
            page_dict["page_rect"] = page.rect
            all_pages_data.append(page_dict)
            
        doc.close()
        return all_pages_data
    except Exception as e:
        print(f"Error processing PDF file {os.path.basename(file_path)}: {e}")
        return None