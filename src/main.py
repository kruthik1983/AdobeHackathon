import os
from pdf_parser import process_pdf
from feature_extractor import extract_features
from heading_classifier import classify_headings
from json_builder import create_json_file

INPUT_DIR = "E:/PDFExtracter/Adobe Hackathon/test"
OUTPUT_DIR = "E:/PDFExtracter/Adobe Hackathon/output"

def main():
    """
    Main orchestration script.
    
    Scans the input directory for PDF files, processes each one through the
    extraction and classification pipeline, and saves the resulting structured
    outline as a JSON file in the output directory.
    """
    print("Starting PDF outline extraction process...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found at {INPUT_DIR}")
        return

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    for filename in pdf_files:
        print(f"\nProcessing file: {filename}...")
        file_path = os.path.join(INPUT_DIR, filename)


        pages_data = process_pdf(file_path)
        if not pages_data:
            print(f"Could not process {filename}. Skipping.")
            continue


        feature_lines = extract_features(pages_data)
        if not feature_lines:
            print(f"No text features extracted from {filename}. Skipping.")
            continue


        title, outline = classify_headings(feature_lines)


        create_json_file(title, outline, filename, OUTPUT_DIR)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()