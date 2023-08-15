
import ast
import re

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

def parse_score(oai_review, logger):
    match = re.search(one_score_pattern, oai_review)
    if not match:
        match = re.search(one_score_pattern_backup, oai_review)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1.0
        logger.error("No rating found in response")
    score = float(rating)
    return score