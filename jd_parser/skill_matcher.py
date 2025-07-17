# jd_parser/skill_matcher.py

import re
import spacy
from sentence_transformers import SentenceTransformer, util
from config.skills import ROLE_BASED_SKILLS, SYNONYM_MAP
from resume_matcher.skill_helpers import normalize_skill  # ✅ FIXED: No circular import

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Build list of all known skills from ROLE_BASED_SKILLS
ALL_KNOWN_SKILLS = sorted({skill for skills in ROLE_BASED_SKILLS.values() for skill in skills})

# Load sentence transformer model for semantic fallback
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# ✅ Clean and normalize text
def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s\.\+\#]", " ", text.lower())
    return text


# ✅ Tokenize and remove stop words
def tokenize(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)


# ✅ Regex-based enrichment for missing HTML/CSS variants
def extract_html_css_variants(text):
    variants = set()
    if re.search(r"\bhtml\s*5\b", text, re.IGNORECASE):
        variants.add("HTML")
    if re.search(r"\bcss\s*3\b", text, re.IGNORECASE):
        variants.add("CSS")
    return variants


# ✅ Semantic fallback matcher using sentence transformers
def semantic_skill_match(text, known_skills, threshold=0.75):

    sentences = text.split("\n")
    doc_embeddings = model.encode(sentences, convert_to_tensor=True, device="cpu")
    skill_embeddings = model.encode(known_skills, convert_to_tensor=True, device="cpu")

    hits = util.cos_sim(doc_embeddings, skill_embeddings)
    matched = set()

    for i in range(len(sentences)):
        for j in range(len(known_skills)):
            if hits[i][j] > threshold:
                matched.add(known_skills[j].title())

    return list(matched)


# ✅ Core matching function: combines exact match, synonym match, and fallback
def match_skills(text, skill_list=None):
    text_clean = preprocess(text)
    tokens = tokenize(text_clean)
    matched = set()

    skills_to_check = skill_list if skill_list else ALL_KNOWN_SKILLS

    # Step 1: Exact match (single and multi-word)
    for skill in skills_to_check:
        norm_skill = normalize_skill(skill)
        if " " in norm_skill:
            if norm_skill in text_clean:
                matched.add(skill)
        else:
            if norm_skill in tokens:
                matched.add(skill)

    # Step 2: Regex pattern for HTML/CSS variants
    regex_variants = extract_html_css_variants(text)
    matched.update(regex_variants)

    # Step 3: Synonym expansion
    for canonical, variants in SYNONYM_MAP.items():
        for variant in variants:
            variant_lower = normalize_skill(variant)
            if (" " in variant_lower and variant_lower in text_clean) or (variant_lower in tokens):
                matched.add(canonical.title())
                break

    # Step 4: Semantic fallback if low match count
    if len(matched) < 3:
        fallback = semantic_skill_match(text, skills_to_check, threshold=0.85)
        matched.update(fallback)

    return sorted(matched)
