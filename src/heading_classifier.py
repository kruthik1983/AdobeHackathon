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

    # Rule 1: Lines that end with a full stop are almost always not headings.
    # This is a very strong signal for body text.
    if text.endswith('.'):
        return True

    # Rule 2: Lines that are too long (more than a typical heading length) are likely paragraphs.
    if len(text.split()) > 10: # Increased strictness for length
        return True
    
    # Rule 3: Very short, non-alphabetic lines (separators, etc.)
    if len(text) > 0 and sum(c.isalpha() for c in text) / (len(text) + 1e-6) < 0.2:
        return True
    
    # Rule 4: Common page artifacts (e.g., page numbers, generic headers/footers)
    if re.match(r"^\s*(page|fig\.|table)\s+\d+\s*$", text_lower): # More specific to avoid filtering valid short headings
        return True
    
    # Rule 5: Generic placeholder text
    if "lorem ipsum" in text_lower:
        return True
        
    return False

def classify_headings(feature_lines):
    """
    Classifies text lines into Title, H1, H2, H3 based on robust feature analysis and adaptive clustering.
    This function implements a more sophisticated heuristic-based approach:
    1. Identifies approximate body text font size using statistical methods (mode/median).
    2. Filters for potential heading candidates based on size, boldness, and structural cues.
    3. Identifies the document title with a weighted heuristic (size, position, uniqueness).
    4. Groups remaining heading candidates by their visual style (font size, boldness, indentation, color).
    5. Dynamically assigns H1, H2, H3 levels by sorting these styles by prominence
       (size, then boldness, then indentation, then color) and mapping them adaptively.
    6. Ensures logical hierarchy (e.g., H2 cannot be larger than H1).
    7. Applies noise filtering to remove non-heading content.
    8. Sorts the final list of headings by their appearance in the document.
    Args:
        feature_lines: A list of feature vectors from feature_extractor.
    Returns:
        A tuple containing the document title (str) and the outline (a list of heading dicts).
    """
    if not feature_lines:
        return "Untitled Document", []

    # --- 1. Identify Body Text Style Robustly ---
    # Collect font sizes from lines likely to be body text (e.g., not extremely large, not single words)
    # Using `round(..., 1)` to group similar font sizes.
    potential_body_sizes = [
        round(line["font_size"], 1) for line in feature_lines
        if line["font_size"] > 5 and line["font_size"] < 25 and line["word_count"] > 4 # Filter out very small/large and short lines
    ]
    
    body_font_size = 10.0 # Default fallback
    if potential_body_sizes:
        # Use the mode for body font size, as it's robust to outliers
        body_font_size_counts = Counter(potential_body_sizes)
        body_font_size = body_font_size_counts.most_common(1)[0][0]

    # --- 2. Identify Title Candidate (Adaptive Page Title Detection WOW Factor) ---
    # Prioritize: Highest Y-position on page 1, then largest font size, then bold/centered.
    title_candidate = {"font_size": 0.0, "text": "Untitled Document", "page_number": 0, "y0": float('inf'), "is_bold": False, "is_centered": False, "font_color": 0}
    
    first_page_lines = [line for line in feature_lines if line["page_number"] == 1]
    
    for line in first_page_lines:
        # Skip very short lines or lines that are clearly not titles
        if line["word_count"] < 2 or len(line["text"].strip()) < 5:
            continue

        # A line is a strong title candidate if:
        # 1. It's higher on the page than the current best candidate.
        # 2. Or, it's at a similar height, but has a larger font size.
        # 3. Or, same height & size, but bolder/more centered.
        # 4. NEW: Consider font color in tie-breaking.

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
                elif line["font_color"] != title_candidate["font_color"] and title_candidate["font_color"] == 0: # Prefer non-black color if current is black
                    is_stronger_candidate = True

        if is_stronger_candidate:
            title_candidate = line.copy()
            # If a line becomes the title candidate, it should definitely not be a heading candidate
            title_candidate["_is_title"] = True
        else:
            line["_is_title"] = False # Mark others

    title_text = title_candidate["text"].replace('\n', ' ').strip()
    
    # --- 3. Filter for Potential Headings (Visual Heuristics Ensemble WOW Factor) ---
    heading_candidates = []
    for line in feature_lines:
        cleaned_text = line["text"].replace('\n', ' ').strip()
        if not cleaned_text or cleaned_text == title_text or line.get("_is_title", False):
            continue
        
        # Apply noise filtering early and aggressively
        if is_noise(cleaned_text):
            continue

        # Heuristic checks for actual heading characteristics
        is_larger_than_body = line["font_size"] > body_font_size * 1.15
        is_bold_and_prominent = line["is_bold"] and line["font_size"] >= body_font_size * 0.95
        is_patterned_heading = line["starts_with_pattern"] and line["word_count"] < 30
        has_significant_space_above = line["space_above"] > (body_font_size * 0.8)
        is_distinct_color = (line["font_color"] != 0 and line["font_color"] != 16777215) # Not black and not white

        # Combine heuristics: A line must meet a strong primary criterion AND not be noise.
        # This makes the classification much stricter.
        if (is_larger_than_body or is_bold_and_prominent or is_patterned_heading or has_significant_space_above or is_distinct_color):
            line["_explanation"] = "Candidate based on font size, bold, pattern, spacing, or color."
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
        # Use rounded font size, boldness, and a representative x0 for style key
        # Also include font_color for a more distinct style key
        style_key = (round(h["font_size"], 1), h["is_bold"], get_closest_representative_x0(h["x0"]), h["font_color"])
        heading_styles[style_key].append(h)

    sorted_styles = sorted(
        heading_styles.keys(), 
        key=lambda s: (s[0], s[1], s[2], s[3]), # s[0]=font_size, s[1]=is_bold, s[2]=x0, s[3]=font_color
        reverse=True # Largest font size first, then bold, then most left, then distinct color
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
        
        # Re-apply noise filtering as a final check before adding to outline
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