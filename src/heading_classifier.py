# src/heading_classifier.py
from collections import Counter, defaultdict
import numpy as np

def classify_headings(feature_lines):
    """
    Classifies text lines into Title, H1, H2, H3 based on robust feature analysis and adaptive clustering.

    This function implements a more sophisticated heuristic-based approach:
    1. Identifies approximate body text font size using statistical methods (mode/median).
    2. Filters for potential heading candidates based on size, boldness, and structural cues.
    3. Identifies the document title with a weighted heuristic (size, position, uniqueness).
    4. Groups remaining heading candidates by their visual style (font size, boldness, indentation).
    5. Dynamically assigns H1, H2, H3 levels by sorting these styles by prominence
       (size, then boldness, then indentation) and mapping them adaptively.
    6. Ensures logical hierarchy (e.g., H2 cannot be larger than H1).
    7. Sorts the final list of headings by their appearance in the document.

    Args:
        feature_lines: A list of feature vectors from feature_extractor.

    Returns:
        A tuple containing the document title (str) and the outline (a list of heading dicts).
    """
    if not feature_lines:
        return "Untitled", []

    # --- 1. Identify Body Text Style Robustly ---
    # Collect all font sizes for body text estimation
    all_font_sizes = [round(line["font_size"], 1) for line in feature_lines]
    
    if not all_font_sizes:
        return "Untitled", []

    # Use the mode for body font size, as it's less sensitive to outliers than mean
    # Filter out very small or very large sizes that are unlikely to be body text
    filtered_font_sizes = [s for s in all_font_sizes if s > 5 and s < 30] # Common range for body text
    if not filtered_font_sizes:
        # Fallback if filtering is too aggressive
        filtered_font_sizes = all_font_sizes 

    # Find the most frequent font size. If ties, Counter.most_common picks one arbitrarily.
    body_font_size_counts = Counter(filtered_font_sizes)
    body_font_size = body_font_size_counts.most_common(1)[0][0] if body_font_size_counts else 10 # Default fallback

    # --- 2. Identify Title Candidate ---
    title_candidate = {"font_size": 0, "text": "Untitled", "page_number": 0, "y0": -1}
    for line in feature_lines:
        # Prioritize larger text on the first page, especially if centered or at the top
        if line["page_number"] == 1:
            # Strong signal if significantly larger than body, near top, and/or centered
            if line["font_size"] > title_candidate["font_size"] * 1.05 and \
               (line["font_size"] > body_font_size * 1.5 or line["is_centered"] or line["y0"] < line["page_height"] * 0.2):
                title_candidate = line
            # If same font size, pick the higher one
            elif line["font_size"] == title_candidate["font_size"] and line["y0"] < title_candidate["y0"]:
                 title_candidate = line

    title_text = title_candidate["text"].replace('\n', ' ').strip()
    
    # --- 3. Filter for Potential Headings ---
    # A line is a heading candidate if:
    # - Its font size is greater than the body font size (with a small margin to catch slight differences).
    # - OR it's bold AND its font size is at least the body font size.
    # - OR it starts with a recognized numbering/bullet pattern and is not excessively long.
    # - It's not the identified title (to avoid duplication).
    # - It's not excessively long (typically headings are concise).
    heading_candidates = []
    for line in feature_lines:
        if line["text"] == title_text: # Skip if it's the title
            continue
        
        is_larger_than_body = line["font_size"] > body_font_size * 1.05
        is_bold_and_at_least_body_size = line["is_bold"] and line["font_size"] >= body_font_size * 0.95
        is_patterned_heading = line["starts_with_pattern"] and line["word_count"] < 30 # Numbered lists are strong cues

        if (is_larger_than_body or is_bold_and_at_least_body_size or is_patterned_heading) and \
           line["word_count"] < 250: # Avoid very long lines that might be body text
            heading_candidates.append(line)

    # --- 4. Group Candidates by Visual Style and Assign Adaptive Levels ---
    # Group by (rounded_font_size, is_bold, normalized_x0)
    # Normalized x0 helps distinguish headings with different indentation levels (e.g., hanging indents vs true subheadings)
    # Quantize x0 to clusters to account for slight variations
    x0_values = [h["x0"] for h in heading_candidates]
    if x0_values:
        # Simple clustering for x0: group values that are close together
        x0_clusters = []
        x0_values.sort()
        for x in x0_values:
            found_cluster = False
            for i, cluster in enumerate(x0_clusters):
                if abs(x - cluster[0]) < 10: # Cluster if within 10 units (arbitrary threshold)
                    x0_clusters[i].append(x)
                    found_cluster = True
                    break
            if not found_cluster:
                x0_clusters.append([x])
        # Use the mean of each cluster as the representative x0
        representative_x0s = sorted([np.mean(c) for c in x0_clusters])
    else:
        representative_x0s = []

    def get_representative_x0(x):
        if not representative_x0s: return x
        # Find the closest representative x0
        return min(representative_x0s, key=lambda rx: abs(rx - x))

    heading_styles = defaultdict(list)
    for h in heading_candidates:
        # Use rounded font size and a representative x0 for style key
        style_key = (round(h["font_size"], 1), h["is_bold"], get_representative_x0(h["x0"]))
        heading_styles[style_key].append(h)

    # Sort styles to define hierarchy:
    # Primary: font_size (descending)
    # Secondary: is_bold (True first, then False)
    # Tertiary: x0 (ascending - left-most is higher level)
    sorted_styles = sorted(
        heading_styles.keys(), 
        key=lambda s: (s[0], s[1], -s[2]), # s[0]=font_size, s[1]=is_bold, s[2]=x0. Sort x0 descending for visual order
        reverse=True # Largest font size first, then bold, then most left
    )

    outline = []
    assigned_levels = {} # To keep track of which style maps to which level (H1, H2, H3)
    
    # Assign H1, H2, H3 dynamically based on distinct prominent styles
    level_counter = 0
    prev_font_size = float('inf')
    prev_x0 = float('inf')

    for style_key in sorted_styles:
        current_font_size, current_is_bold, current_x0 = style_key
        
        # Heuristic for new level:
        # 1. Significantly different (smaller) font size OR
        # 2. Same font size but no longer bold AND previous was bold OR
        # 3. Same font size, same bold, but significant indentation (x0 changed to the right)
        
        # Only assign up to H3
        if level_counter < 3:
            # Check for significant font size drop or boldness change for a new primary level
            if current_font_size < prev_font_size * 0.90 or \
               (current_font_size == prev_font_size and not current_is_bold and assigned_levels.get(prev_font_size, {}).get(prev_x0) is True) or \
               (current_font_size == prev_font_size and current_is_bold == assigned_levels.get(prev_font_size, {}).get(prev_x0, False) and current_x0 > prev_x0 + 10): # Indentation change
                level_counter += 1
            
            # Map the current style key to the determined level
            if level_counter < 3: # Ensure we don't go beyond H3
                assigned_levels[style_key] = f"H{level_counter + 1}"
            else:
                assigned_levels[style_key] = "H3" # Cap at H3
        else:
            assigned_levels[style_key] = "H3" # Any further distinct styles are also H3 for now

        prev_font_size = current_font_size
        prev_x0 = current_x0


    # If no levels were assigned (e.g., all lines are body or only one heading type)
    if not assigned_levels and heading_candidates:
        assigned_levels[sorted_styles[0]] = "H1" # Default the most prominent to H1
        if len(sorted_styles) > 1:
            assigned_levels[sorted_styles[1]] = "H2"
        if len(sorted_styles) > 2:
            assigned_levels[sorted_styles[2]] = "H3"


    # Build the final outline using assigned levels
    for h in heading_candidates:
        style_key = (round(h["font_size"], 1), h["is_bold"], get_representative_x0(h["x0"]))
        level = assigned_levels.get(style_key)
        
        # Fallback if a candidate style didn't get assigned a level (shouldn't happen with logic above)
        if level is None:
            # This can happen if a document only has a few similar "heading-like" lines.
            # Assign based on simple size comparison relative to the largest heading style found
            if assigned_levels:
                max_heading_font_size = sorted_styles[0][0]
                if h["font_size"] >= max_heading_font_size * 0.95:
                    level = "H1"
                elif h["font_size"] >= body_font_size * 1.5:
                    level = "H2"
                else:
                    level = "H3"
            else:
                level = "H1" # Default if no headings found


        outline.append({
            "level": level,
            "text": h["text"].replace('\n', ' ').strip(), # Clean text
            "page": h["page_number"],
            "y0": h["y0"] # Keep y0 for final sort
        })

    # Final sort by page number and then vertical position (y0)
    outline.sort(key=lambda x: (x["page"], x["y0"]))

    # Remove temporary y0 field
    for item in outline:
        del item["y0"]

    return title_text, outline