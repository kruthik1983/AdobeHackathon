import os
import sys
import time
import argparse
import pandas as pd

from pdf_parser import process_pdf
from feature_extractor import extract_features
from heading_classifier_ml import classify_headings, train_and_save_model
from json_builder import create_json_file
from font_analysis_logger import log_font_styles

def create_dataset_mode(input_dir: str, dataset_dir: str):
    """
    Processes all PDFs in the input directory to create a single, comprehensive
    CSV file with all extracted features, ready for labeling.
    """
    print("--- Running in Dataset Creation Mode ---")
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    all_features_list = []
    print(f"Found {len(pdf_files)} PDF(s) to process for dataset creation.")

    for pdf_filename in pdf_files:
        file_path = os.path.join(input_dir, pdf_filename)
        print(f"  - Extracting features from '{pdf_filename}'...")
        pages_data = process_pdf(file_path)
        if not pages_data:
            continue
        current_doc_features = extract_features(pages_data)
        for feature in current_doc_features:
            feature['source_filename'] = pdf_filename
        all_features_list.extend(current_doc_features)

    if not all_features_list:
        print("No features were extracted from any PDF files.")
        return

    df = pd.DataFrame(all_features_list)
    df['level'] = '' 

    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, 'pdf_analyzer.csv')
    df.to_csv(dataset_path, index=False, encoding='utf-8')

    print(f"\nSuccessfully created dataset with {len(df)} rows.")
    print(f"Dataset saved to: '{dataset_path}'")
    print("Next step: Open this CSV file and fill in the 'level' column with labels (e.g., Title, H1, Body).")

def classify_mode(input_dir: str, output_dir: str, model_dir: str):
    """
    Default mode: Orchestrates the PDF outline extraction process using a pre-trained ML model.
    """
    print("--- Running in Classification Mode ---")
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    for pdf_filename in pdf_files:
        start_time = time.time()
        file_path = os.path.join(input_dir, pdf_filename)
        print(f"\n--- Processing '{pdf_filename}' ---")

        pages_data = process_pdf(file_path)
        if not pages_data:
            create_json_file("", [], pdf_filename, output_dir)
            continue

        current_doc_features = extract_features(pages_data)
        if not current_doc_features:
            create_json_file("", [], pdf_filename, output_dir)
            continue

        title, outline = classify_headings(current_doc_features, model_dir)
        create_json_file(title, outline, pdf_filename, output_dir)

        processing_time = time.time() - start_time
        print(f"Finished processing '{pdf_filename}' in {processing_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="PDF Outline Extraction Tool")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['classify', 'dataset'],
        default='classify',
        help="Operation mode: 'classify' to generate JSON outlines, 'dataset' to create a CSV for training."
    )
    args = parser.parse_args()

    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "input")
    output_dir = os.path.join(current_dir, "output")
    model_dir = os.path.join(current_dir, "model")
    dataset_dir = os.path.join(current_dir, "dataset")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found. Please place PDFs in '{input_dir}'.")
        sys.exit(1)

    if args.mode == 'dataset':
        create_dataset_mode(input_dir, dataset_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
        classify_mode(input_dir, output_dir, model_dir)

if __name__ == "__main__":
    main()
