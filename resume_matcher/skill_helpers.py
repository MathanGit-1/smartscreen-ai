import re
# resume_matcher/skill_helpers.pyimport re
from config.skills import SYNONYM_MAP, SPECIAL_CHARACTER_SKILLS

def normalize_skill(text):
    raw_text = text.lower().strip()
    
    # Handle special character skills directly (e.g., c#, c++, f#)
    if raw_text in SPECIAL_CHARACTER_SKILLS:
        return raw_text

    # Strip everything except alphanumerics (so css 3 → css3, javascript 4+ → javascript4)
    clean_text = re.sub(r'[^a-z0-9]+', '', raw_text)

    # Compare against synonym variants
    for canonical, variants in SYNONYM_MAP.items():
        norm_canonical = re.sub(r'[^a-z0-9]+', '', canonical.lower())
        norm_variants = [re.sub(r'[^a-z0-9]+', '', v.lower()) for v in variants]
        if clean_text == norm_canonical or clean_text in norm_variants:
            return canonical

    return clean_text


def apply_reverse_synonyms(skills):
    return list({normalize_skill(skill) for skill in skills})

def expand_synonyms(skills):
    expanded = set(skills)
    for skill in skills:
        norm = normalize_skill(skill)
        for key, variants in SYNONYM_MAP.items():
            if norm == normalize_skill(key):
                expanded.update(variants)
    return list(expanded)