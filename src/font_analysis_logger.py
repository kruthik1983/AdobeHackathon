import csv
import os
from collections import defaultdict, Counter

def log_font_styles(feature_lines: list, output_dir: str, filename_prefix: str = "font_styles_report"):
    """
    Logs unique font styles (size, name, bold, italic, centered) and their character counts found in the document.
    This helps in understanding the document's typography and can be a "professional/debuggable" WOW factor.
    """
    font_style_data = defaultdict(lambda: {'char_count': 0, 'line_count': 0, 'sample_text': []})
    
    for line in feature_lines:
        style_key = (
            round(line["font_size"], 1),
            line["font_name"],
            line["is_bold"],
            line["is_italic"],
            line["is_centered"],
            round(line["x0"], 0),
            line["font_color"]
        )
        font_style_data[style_key]['char_count'] += line["text_length"] # Count by characters for dominance
        font_style_data[style_key]['line_count'] += 1
        if len(font_style_data[style_key]['sample_text']) < 5: # Store a few samples
            font_style_data[style_key]['sample_text'].append(line["text"][:70].replace('\n', ' ') + "..." if len(line["text"]) > 70 else line["text"].replace('\n', ' '))

    output_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Font_Size', 'Font_Name', 'Is_Bold', 'Is_Italic', 'Is_Centered', 'X_Position_Approx', 'Font_Color', 'Total_Chars', 'Total_Lines', 'Sample_Text_Lines']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
            sorted_styles = sorted(font_style_data.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]), reverse=True)

            for style_key, data in sorted_styles:
                writer.writerow({
                    'Font_Size': style_key[0],
                    'Font_Name': style_key[1],
                    'Is_Bold': style_key[2],
                    'Is_Italic': style_key[3],
                    'Is_Centered': style_key[4],
                    'X_Position_Approx': style_key[5],
                    'Font_Color': style_key[6], 
                    'Total_Chars': data['char_count'],
                    'Total_Lines': data['line_count'],
                    'Sample_Text_Lines': " | ".join(data['sample_text'])
                })
        print(f"Logged font styles to: {output_path}")
    except Exception as e:
        print(f"Error logging font styles: {e}")