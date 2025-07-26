# src/header_footer_detector.py
from collections import defaultdict, Counter

def get_header_footer_candidates(feature_lines: list, page_count: int, min_repetition: float = 0.5):
    """
    Identifies text lines that are likely headers or footers based on their consistent position
    and repetition across multiple pages.
    Args:
        feature_lines: A list of feature vectors for all text lines in the document.
        page_count: The total number of pages in the document.
        min_repetition: The minimum fraction of pages a line must appear on to be considered
                        a header/footer candidate.
    Returns:
        A list of text strings that are likely headers or footers.
    """
    if page_count < 2:
        return []

    line_counts = defaultdict(Counter)
    line_positions = defaultdict(list)

    # Aggregate line data by text and position (y0)
    for line in feature_lines:
        text_normalized = line["text"].strip().lower()
        if not text_normalized or len(text_normalized) < 5 or line["word_count"] > 10:
            continue
        
        # Use a rounded y0 position to group lines that appear at the same height on different pages
        pos_key = round(line["y0"], -1) # Round to nearest 10 for tolerance
        line_counts[text_normalized][pos_key] += 1
        line_positions[text_normalized].append(pos_key)
        
    header_footer_candidates = set()

    # Determine the minimum count for a line to be considered repetitive
    min_count = page_count * min_repetition

    for text, counts in line_counts.items():
        for pos, count in counts.items():
            if count >= min_count:
                # Add to candidates. Using the original text for the filter.
                header_footer_candidates.add(text)

    return list(header_footer_candidates)