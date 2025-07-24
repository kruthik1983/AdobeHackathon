# src/main.py
import os
import sys
import time # For performance measurement

# Import functions from other modules
from .pdf_parser import process_pdf
from .feature_extractor import extract_features
from .heading_classifier import classify_headings
from .json_builder import create_json_file

def main():
    """
    Orchestrates the PDF outline extraction process.
    Reads PDFs from /app/input, processes them, and writes JSON outlines to /app/output.
    """
    input_dir = "/app/input"
    output_dir = "/app/output"

    # Ensure input directory exists for the container to function
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found. Please mount your input PDFs to '{input_dir}' using docker run -v.")
        sys.exit(1)

    # Ensure output directory exists (Docker will create it, but good practice)
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'. Please place PDFs in this directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process in '{input_dir}'.")

    for pdf_filename in pdf_files:
        start_time = time.time()
        file_path = os.path.join(input_dir, pdf_filename)
        print(f"\n--- Processing '{pdf_filename}' ---")

        # 1. Parse PDF to extract raw structured data
        pages_data = process_pdf(file_path)
        if pages_data is None or not pages_data:
            print(f"Skipping '{pdf_filename}' due to parsing error or no content.")
            create_json_file("Untitled Document", [], pdf_filename, output_dir) # Output empty outline
            continue

        # 2. Extract features from each text line
        all_lines_features = extract_features(pages_data)
        if not all_lines_features:
            print(f"No text lines or features extracted from '{pdf_filename}'.")
            create_json_file("Untitled Document", [], pdf_filename, output_dir) # Output empty outline
            continue

        # 3. Classify headings and determine the document title
        title, outline = classify_headings(all_lines_features)
        
        # If title is empty/generic after classification, try to derive from first H1 or filename
        if not title or title.strip().lower() in ["untitled", "untitled document"]:
            if outline:
                title = outline[0]["text"] # Use first H1 as title if available
            else:
                title = os.path.splitext(pdf_filename)[0].replace('_', ' ').replace('-', ' ').title() # Fallback to cleaned filename
        
        # 4. Build and write the final JSON output
        create_json_file(title, outline, pdf_filename, output_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Finished processing '{pdf_filename}' in {processing_time:.2f} seconds.")

if __name__ == "__main__":
    main()