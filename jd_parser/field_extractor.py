import re
import spacy
import json

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load unified config file with labels + patterns
with open("config/field_config.json", "r") as f:
    FIELD_CONFIG = json.load(f)

# 🔍 Hybrid extraction function
def extract_field(text: str, labels: list, patterns: list):
    for line in text.split("\n"):
        for kw in labels:
            if kw.lower() in line.lower():
                for pattern in patterns:
                    match = re.search(pattern, line.strip(), re.IGNORECASE)
                    if match:
                        return _extract_value(match)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _extract_value(match)
    return None

def _extract_value(match):
    if match.lastindex:
        return " - ".join([g for g in match.groups() if g])
    return match.group(0).strip()

# 🎯 Role extraction with fallback
def extract_role(text):
    role_line = extract_field(text, FIELD_CONFIG["role"]["labels"], FIELD_CONFIG["role"]["patterns"])
    if role_line:
        return role_line.strip().title()

    # spaCy fallback (Job Title entities)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "JOB_TITLE":
            return ent.text.title()

    return None

# 🧠 Main extraction function
def extract_fields_from_text(text):
    jd_id = "JD_" + str(abs(hash(text)))[0:6]

    role = extract_role(text)

    yoe = extract_field(text, FIELD_CONFIG["yoe"]["labels"], FIELD_CONFIG["yoe"]["patterns"])
    notice_period = extract_field(text, FIELD_CONFIG["notice_period"]["labels"], FIELD_CONFIG["notice_period"]["patterns"])
    num_positions = extract_field(text, FIELD_CONFIG["num_positions"]["labels"], FIELD_CONFIG["num_positions"]["patterns"])
    work_location = extract_field(text, FIELD_CONFIG["work_location"]["labels"], FIELD_CONFIG["work_location"]["patterns"])
    shift_timing = extract_field(text, FIELD_CONFIG["shift_timing"]["labels"], FIELD_CONFIG["shift_timing"]["patterns"])

    return {
        "jd_id": jd_id,
        "role": role,
        "yoe": yoe,
        "notice_period": notice_period,
        "num_positions": num_positions,
        "work_location": work_location,
        "shift_timing": shift_timing
    }
