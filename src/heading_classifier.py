import numpy as np
import re
from collections import Counter, defaultdict
import fitz # Import fitz to recognize fitz.Rect objects

def is_noise(text: str) -> bool:
    """
    Filters out common noise patterns for lines that are likely body text,
    even if they have some formatting. This is crucial for avoiding
    misclassification of paragraphs/list items as headings.
    """
    text_lower = text.lower().strip()

    # Rule 1: Lines that end with a full stop, question mark, or exclamation mark
    # are almost always not headings. This is a very strong signal for body text.
    if text.endswith('.') or text.endswith('?') or text.endswith('!'):
        return True

    # Rule 2: Lines that are too long (more than a typical heading length) are likely paragraphs.
    # Headings are usually concise.
    if len(text.split()) > 12: # Increased strictness for length
        return True
    
    # Rule 3: Very short, non-alphabetic lines (separators, page numbers, etc.)
    # This catches lines like "---" or "Page 1".
    if len(text) > 0 and sum(c.isalpha() for c in text) / (len(text) + 1e-6) < 0.2:
        return True
    
    # Rule 4: Common page artifacts (e.g., generic headers/footers that might be bolded)
    # Be specific to avoid filtering valid short headings.
    if re.match(r"^\s*(page|fig\.|table)\s+\d+\s*$", text_lower):
        return True
    
    # Rule 5: Generic placeholder text
    if "lorem ipsum" in text_lower:
        return True
    
    # Rule 6: Common list item patterns that are short but not meaningful headings
    if re.match(r"^\s*([\u2022\u25CF\u25BA]|\d+\.)$", text_lower) and len(text.split()) < 3:
        return True
        
    return False

def classify_headings(feature_lines, noisy_patterns: list):
    """
    Classifies text lines into Title, H1, H2, H3 based on robust feature analysis and adaptive clustering.
    This function implements a more sophisticated heuristic-based approach:
    1. Identifies approximate body text font size and color.
    2. Filters out noisy header/footer text identified by the detector.
    3. Filters for potential heading candidates based on strong visual cues.
    4. Identifies the document title with a weighted heuristic (size, position, uniqueness).
    5. Groups remaining heading candidates by their distinct visual style (font size, boldness, indentation, color).
    6. Dynamically assigns H1, H2, H3 levels by sorting these styles by prominence
       and mapping them adaptively, prioritizing clear visual breaks.
    7. Applies aggressive noise filtering to remove non-heading content.
    8. Sorts the final list of headings by their appearance in the document.
    Args:
        feature_lines: A list of feature vectors from feature_extractor.
        noisy_patterns: A list of strings to be filtered out (from header/footer detector).
    Returns:
        A tuple containing the document title (str) and the outline (a list of heading dicts).
    """
    # Filter out noisy patterns first
    filtered_lines = [line for line in feature_lines if line["text"].strip().lower() not in noisy_patterns]
    
    if not filtered_lines:
        return "Untitled Document", []

    # --- 1. Identify Body Text Style Robustly ---
    # Collect font sizes and colors from lines likely to be body text.
    # Filter out very small/large fonts and very short lines which are unlikely to be body.
    potential_body_lines = [
        line for line in filtered_lines
        if line["font_size"] > 5 and line["font_size"] < 25 and line["word_count"] > 4 and not line["is_bold"]
    ]
    
    body_font_size = 10.0 # Default fallback
    body_font_color = 0 # Default to black
    if potential_body_lines:
        # Get mode font size
        body_font_size_counts = Counter([round(line["font_size"], 1) for line in potential_body_lines])
        if body_font_size_counts:
            body_font_size = body_font_size_counts.most_common(1)[0][0]
        
        # Get mode font color
        body_font_color_counts = Counter([line["font_color"] for line in potential_body_lines])
        if body_font_color_counts:
            body_font_color = body_font_color_counts.most_common(1)[0][0]

    # --- 2. Identify Title Candidate (Adaptive Page Title Detection WOW Factor) ---
    # Prioritize: Highest Y-position on page 1, then largest font size, then bold/centered/distinct color.
    title_candidate = {"font_size": 0.0, "text": "Untitled Document", "page_number": 0, "y0": float('inf'), "is_bold": False, "is_centered": False, "font_color": 0}
    
    first_page_lines = [line for line in filtered_lines if line["page_number"] == 1]
    
    for line in first_page_lines:
        # Skip very short lines or lines that are clearly not titles
        if is_noise(line["text"]) or line["word_count"] < 2 or len(line["text"].strip()) < 5:
            continue

        is_stronger_candidate = False

        if line["y0"] < title_candidate["y0"] - (body_font_size * 0.5): # Significantly higher
            is_stronger_candidate = True
        elif abs(line["y0"] - title_candidate["y0"]) < (body_font_size * 0.5): # Similar height
            if line["font_size"] > title_candidate["font_size"] * 1.05: # Significantly larger font
                is_stronger_candidate = True
            elif line["font_size"] == title_candidate["font_size"]:
                if line["is_bold"] and not title_candidate["is_bold"]: # Same size/height, but current is bold
                    is_stronger_candidate = True
                elif line["is_centered"] and not title_candidate["is_centered"]: # Same size/height/bold, but current is centered
                    is_stronger_candidate = True
                # Prefer lines with a distinct color if the current candidate is black (0)
                elif line["font_color"] != body_font_color and title_candidate["font_color"] == body_font_color:
                    is_stronger_candidate = True

        if is_stronger_candidate:
            title_candidate = line.copy()
            title_candidate["_is_title"] = True # Mark as title
        else:
            line["_is_title"] = False # Mark others

    title_text = title_candidate["text"].replace('\n', ' ').strip()
    
    # --- 3. Filter for Potential Headings (Visual Heuristics Ensemble WOW Factor) ---
    heading_candidates = []
    for line in filtered_lines:
        cleaned_text = line["text"].replace('\n', ' ').strip()
        if not cleaned_text or cleaned_text == title_text or line.get("_is_title", False):
            continue
        
        if is_noise(cleaned_text):
            continue

        is_significantly_larger = line["font_size"] > body_font_size * 1.15
        is_bold_and_distinct_color = line["is_bold"] and line["font_color"] != body_font_color
        is_bold_and_prominent_size = line["is_bold"] and line["font_size"] >= body_font_size * 1.05
        has_significant_space_above = line["space_above"] > (body_font_size * 0.8)
        starts_with_pattern_and_short = line["starts_with_pattern"] and line["word_count"] < 10

        is_strong_candidate = (
            is_significantly_larger or
            is_bold_and_distinct_color or
            is_bold_and_prominent_size or
            has_significant_space_above or
            starts_with_pattern_and_short
        )
        
        if is_strong_candidate:
            if len(line["text"].split()) > 15 and line["is_bold"] and line["font_color"] == body_font_color:
                continue
            
            line["_explanation"] = "Candidate based on strong visual cues."
            heading_candidates.append(line)

    # --- 4. Group Candidates by Visual Style and Assign Adaptive Levels ---
    x0_values = [h["x0"] for h in heading_candidates]
    representative_x0s = []
    if x0_values:
        x0_values.sort()
        if x0_values:
            current_cluster_start = x0_values[0]
            representative_x0s.append(current_cluster_start)
            for x in x0_values:
                if x - current_cluster_start > 15:
                    current_cluster_start = x
                    representative_x0s.append(current_cluster_start)
    
    def get_closest_representative_x0(x):
        if not representative_x0s: return x
        return min(representative_x0s, key=lambda rx: abs(rx - x))

    heading_styles = defaultdict(list)
    for h in heading_candidates:
        style_key = (round(h["font_size"], 1), h["is_bold"], get_closest_representative_x0(h["x0"]), h["font_color"])
        heading_styles[style_key].append(h)

    sorted_styles = sorted(
        heading_styles.keys(), 
        key=lambda s: (s[0], s[1], s[2], s[3]),
        reverse=True
    )

    outline = []
    assigned_levels_map = {}
    
    current_level_idx = 0
    prev_style_key = None

    for style_key in sorted_styles:
        current_font_size, current_is_bold, current_x0, current_font_color = style_key

        if current_level_idx < 3:
            if prev_style_key is None:
                assigned_levels_map[style_key] = f"H{current_level_idx + 1}"
            else:
                prev_font_size, prev_is_bold, prev_x0, prev_font_color = prev_style_key
                
                font_size_drop = (current_font_size < prev_font_size * 0.90)
                boldness_change = (current_font_size == prev_font_size and prev_is_bold and not current_is_bold)
                indentation_change = (current_font_size == prev_font_size and current_is_bold == prev_is_bold and current_x0 > prev_x0 + 10)
                color_change = (current_font_size == prev_font_size and current_is_bold == prev_is_bold and current_x0 == prev_x0 and current_font_color != prev_font_color)

                if font_size_drop or boldness_change or indentation_change or color_change:
                    current_level_idx += 1
                    if current_level_idx < 3:
                        assigned_levels_map[style_key] = f"H{current_level_idx + 1}"
                    else:
                        assigned_levels_map[style_key] = "H3"
                else:
                    assigned_levels_map[style_key] = assigned_levels_map[prev_style_key]
        else:
            assigned_levels_map[style_key] = "H3"

        prev_style_key = style_key
        
    for h in heading_candidates:
        style_key = (round(h["font_size"], 1), h["is_bold"], get_closest_representative_x0(h["x0"]), h["font_color"])
        heading_level = assigned_levels_map.get(style_key, "H3")
        
        if is_noise(h["text"]):
            continue

        outline_item = {
            "level": heading_level,
            "text": h["text"].replace('\n', ' ').strip(),
            "page": h["page_number"],
            "y0": h["y0"],
        }
        
        outline.append(outline_item)

    outline.sort(key=lambda x: (x["page"], x["y0"]))

    for item in outline:
        del item["y0"]

    return title_text, outline