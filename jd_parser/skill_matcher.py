# jd_parser/skill_matcher.py

import json
import re
import spacy
from sentence_transformers import SentenceTransformer, util

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load skill config
with open("config/skills.json", "r") as f:
    config = json.load(f)

KNOWN_SKILLS = config["skills"]
SKILL_SYNONYMS = config["synonyms"]

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# ✅ Clean and normalize text
def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s\.\+\#]", " ", text.lower())
    return text

# ✅ spaCy tokenizer for processing and removing stop words
def tokenize(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)

# ✅ Semantic match fallback
def semantic_skill_match(text, known_skills, threshold=0.75):
    sentences = text.split("\n")
    doc_embeddings = model.encode(sentences, convert_to_tensor=True)
    skill_embeddings = model.encode(known_skills, convert_to_tensor=True)

    hits = util.cos_sim(doc_embeddings, skill_embeddings)
    matched = set()

    for i in range(len(sentences)):
        for j in range(len(known_skills)):
            if hits[i][j] > threshold:
                matched.add(known_skills[j].title())
    return list(matched)

# ✅ Main matcher function to extract relevant skills from raw text (JD or resume)
def match_skills(text):
    # Step 1: Clean the text using regex (remove special chars, lowercase)
    text_clean = preprocess(text)

    # Step 2: Tokenize the cleaned text using spaCy (with stopword and punctuation removal)
    tokens = tokenize(text_clean)

    # Initialize an empty set to store all matched skills
    matched = set()

    # 1️⃣ Direct skill matching (Exact + Synonym match)
    for skill in KNOWN_SKILLS:
        skill_lower = skill.lower()
        if " " in skill_lower:
            if skill_lower in text_clean:
                matched.add(skill)
        else:
            if skill_lower in tokens:
                matched.add(skill)

    for canonical, variants in SKILL_SYNONYMS.items():
        for variant in variants:
            variant_lower = variant.lower()
            if (" " in variant_lower and variant_lower in text_clean) or (variant_lower in tokens):
                matched.add(canonical.title())
                break

    # 2️⃣ Semantic fallback (only if 2 or fewer skills found)
    if len(matched) < 3:
        semantic_matches = semantic_skill_match(text, KNOWN_SKILLS, threshold=0.85)
        matched.update(semantic_matches)

    return sorted(matched)
