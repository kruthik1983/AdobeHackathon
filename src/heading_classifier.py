from collections import Counter, defaultdict

def classify_headings(feature_lines):
    """
    Classifies text lines into Title, H1, H2, H3 based on feature vectors.

    This function implements the core logic of the headinTupleg detection. It uses
    an unsupervised, heuristic-based approach:
    1. Find the most common font style (size and name) and designate it as body text.
    2. Filter out body text to create a list of heading candidates.
    3. Identify the document title (usually the largest text on the first page).
    4. Group remaining heading candidates by style and rank these styles by prominence
       (font size, boldness) to determine H1, H2, H3 levels.
    5. Sort the final list of headings by their appearance in the document.

    Args:
        feature_lines: A list of feature vectors from feature_extractor.

    Returns:
        A tuple containing the document title (str) and the outline (a list of heading dicts).
    """
    if not feature_lines:
        return "Untitled",

    # 1. Identify body text style by finding the most common style based on character count
    style_char_counts = Counter()
    for line in feature_lines:
        style_key = (round(line["font_size"]), line["font_name"])
        style_char_counts[style_key] += line["text_length"]

    if not style_char_counts:
        return "Untitled",
        
    body_style = style_char_counts.most_common(1)
    body_font_size = body_style[0][0][0]

    print(body_font_size)

    # 2. Filter for potential headings and identify the title
    heading_candidates = []
    title_candidate = {"font_size": 0, "text": "Untitled"}

    for line in feature_lines:
        is_candidate = round(line["font_size"]) > body_font_size and line["word_count"] < 25
        if is_candidate:
            heading_candidates.append(line)
        
        if line["page_number"] == 1 and line["font_size"] > title_candidate["font_size"]:
            title_candidate = line

    title_text = title_candidate["text"]
    heading_candidates = [h for h in heading_candidates if h["text"]!= title_text]

    # 3. Group remaining candidates by style and rank the styles
    heading_styles = defaultdict(list)
    for h in heading_candidates:
        style_key = (round(h["font_size"]), h["is_bold"])
        heading_styles[style_key].append(h)

    # Sort styles: primary key font_size (desc), secondary key is_bold (True first)
    sorted_styles = sorted(heading_styles.keys(), key=lambda s: (s, s[1]), reverse=True)

    # 4. Assign H1, H2, H3 labels based on rank
    level_map = ["H1", "H2", "H3"]
    outline = []
    for i, style in enumerate(sorted_styles):
        if i < len(level_map):
            level = level_map[i]
            for block in heading_styles[style]:
                outline.append({
                    "level": level,
                    "text": block["text"],
                    "page": block["page_number"],
                    "bbox_y0": block["bbox"].y0
                })

    # 5. Final sort by page number and then vertical position
    outline.sort(key=lambda x: (x["page"], x["bbox_y0"]))

    for item in outline:
        del item["bbox_y0"]

    return title_text, outline