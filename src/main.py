import os
import sys
import time
import shutil

# Import functions from other modules
from pdf_parser import process_pdf
from feature_extractor import extract_features
from heading_classifier_ml import classify_headings 
from json_builder import create_json_file
from font_analysis_logger import log_font_styles

def main():
    """
    Orchestrates the PDF outline extraction process using a pre-trained ML model.
    Reads PDFs from /input, processes them, and writes JSON outlines to /output.
    """
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "input")
    output_dir = os.path.join(current_dir, "output")
    
    # --- CHANGE START ---
    # Define the directory where the saved model is located.
    model_dir = os.path.join(current_dir, "model")
    # --- CHANGE END ---

    print(f"Input directory: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found. Please place PDFs in '{input_dir}'.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True) 

    # Clean the output directory before processing
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.islink(file_path) or os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not remove old output item {file_path}: {e}")
    print(f"Cleaned contents of output directory: {output_dir}")

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process in '{input_dir}'.")

    all_docs_features_for_logging = []

    for pdf_filename in pdf_files:
        start_time = time.time()
        file_path = os.path.join(input_dir, pdf_filename)
        print(f"\n--- Processing '{pdf_filename}' ---")

        # 1. Parse PDF
        pages_data = process_pdf(file_path)
        if not pages_data:
            print(f"Skipping '{pdf_filename}' due to parsing error or no content.")
            create_json_file("Untitled Document", [], pdf_filename, output_dir)
            continue

        # 2. Extract features
        current_doc_features = extract_features(pages_data)
        if not current_doc_features:
            print(f"No text lines or features extracted from '{pdf_filename}'.")
            create_json_file("Untitled Document", [], pdf_filename, output_dir)
            continue
        
        # Add source filename for logging purposes
        for feature in current_doc_features:
            feature['source_filename'] = pdf_filename
        all_docs_features_for_logging.extend(current_doc_features)

        # --- CHANGE START ---
        # 3. Classify headings using the pre-trained model
        #    We now pass the 'model_dir' to the function.
        title, outline = classify_headings(current_doc_features, model_dir)
        # --- CHANGE END ---
        
        # 4. Build and write JSON
        create_json_file(title, outline, pdf_filename, output_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Finished processing '{pdf_filename}' in {processing_time:.2f} seconds.")

    # 5. Log font analysis across all processed documents
    if all_docs_features_for_logging:
        log_font_styles(all_docs_features_for_logging, output_dir, filename_prefix="all_docs_font_report")
    else:
        print("No features extracted across all documents to log font styles.")

if __name__ == "__main__":
    main()