import json
import os

def create_json_file(title: str, outline: list, input_filename: str, output_dir: str):
    """
    Constructs the final JSON object in the required format and writes it to a file.
    Args:
        title: The document title.
        outline: The list of classified headings.
        input_filename: The base name of the input PDF file (e.g., "sample.pdf").
        output_dir: The directory to write the JSON file to.
    """
    output_data = {
        "title": title,
        "outline": outline
    }

    # Ensure the output filename has a .json extension
    base_name_without_ext = os.path.splitext(input_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name_without_ext}.json")

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully created JSON outline: {output_path}")
    except Exception as e:
        print(f"Error writing JSON file for {input_filename}: {e}")