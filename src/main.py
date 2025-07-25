import os
import sys
import time # For performance measurement
import shutil # For clearing output directory

# Import functions from other modules
# Use relative imports, as this file will be run as a module (python -m src.main)
from .pdf_parser import process_pdf
from .feature_extractor import extract_features
from .heading_classifier import classify_headings
from .json_builder import create_json_file
from .font_analysis_logger import log_font_styles

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

    # --- START OF MODIFIED OUTPUT DIRECTORY CLEANUP LOGIC ---
    # Ensure output directory exists. If it does, clear its *contents*
    # instead of trying to delete and recreate the directory itself.
    # This avoids "Device or resource busy" errors on mounted volumes.
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory structure exists

    # Iterate and remove contents
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.islink(file_path) or os.path.isfile(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectories recursively
            print(f"Removed old output item: {file_path}")
        except Exception as e:
            # Print an error but do not exit, allowing processing to continue if possible
            print(f"Warning: Could not remove old output item {file_path}: {e}")
    print(f"Cleaned contents of output directory: {output_dir}")
    # --- END OF MODIFIED OUTPUT DIRECTORY CLEANUP LOGIC ---


    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'. Please place PDFs in this directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process in '{input_dir}'.")

    # Collect features from all documents for the combined font analysis report
    all_docs_features_for_logging = []

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
        current_doc_features = extract_features(pages_data)
        if not current_doc_features:
            print(f"No text lines or features extracted from '{pdf_filename}'.")
            create_json_file("Untitled Document", [], pdf_filename, output_dir) # Output empty outline
            continue
        
        # Add a source_filename tag to features for the logger, useful for multi-doc analysis
        for feature in current_doc_features:
            feature['source_filename'] = pdf_filename
        all_docs_features_for_logging.extend(current_doc_features)


        # 3. Classify headings and determine the document title
        title, outline = classify_headings(current_doc_features)
        
        # If title is empty/generic after classification, try to derive from first H1 or filename
        if not title or title.strip().lower() in ["untitled", "untitled document"]:
            if outline:
                # Use the highest-level heading found as a fallback for the title
                highest_level_heading = next((h for h in outline if h['level'] == 'H1'), 
                                             next((h for h in outline if h['level'] == 'H2'), 
                                                  next((h for h in outline if h['level'] == 'H3'), None)))
                if highest_level_heading:
                    title = highest_level_heading["text"]
                else:
                    title = os.path.splitext(pdf_filename)[0].replace('_', ' ').replace('-', ' ').title() # Fallback to cleaned filename
            else:
                title = os.path.splitext(pdf_filename)[0].replace('_', ' ').replace('-', ' ').title() # Fallback to cleaned filename
        
        # 4. Build and write the final JSON output
        create_json_file(title, outline, pdf_filename, output_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Finished processing '{pdf_filename}' in {processing_time:.2f} seconds.")

    # 5. Log comprehensive font analysis for all documents (WOW factor: PDF Fonts & Styles Explorer)
    if all_docs_features_for_logging:
        log_font_styles(all_docs_features_for_logging, output_dir, filename_prefix="all_docs_font_report")
    else:
        print("No features extracted across all documents to log font styles.")

if __name__ == "__main__":
    main()