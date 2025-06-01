from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
from resume_matcher.utils import extract_mobile, extract_email, clean_skills
from jd_parser.skill_matcher import match_skills
import spacy
import time

# Load spaCy NLP model for fallback extraction
nlp = spacy.load("en_core_web_sm")

# ✅ Load model globally (not inside any function)
try:
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
except RuntimeError as e:
    print("⚠️ CUDA failed — likely unsupported GPU. Using CPU instead.")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# Expand synonyms for resume skill variants
SYNONYM_MAP = {
    "llms": ["genai", "large language models", "openai", "chatbot"],
    "hugging face": ["transformers", "open source llm"],
    "rest": ["rest api", "web api"],
    "pytorch": ["torch"],
    "sql": ["tsql", "rdbms"],
}

def expand_synonyms(skills):
    expanded = set(skills)
    for skill in skills:
        for key, synonyms in SYNONYM_MAP.items():
            if skill.lower() == key or key in skill.lower():
                expanded.update(synonyms)
    return list(expanded)

# JD extraction (rule-based + fallback to noun chunks)
def extract_jd_skills(text, min_skills=5):
    skill_keywords = set(clean_skills(match_skills(text)))
    if len(skill_keywords) >= min_skills:
        return list(skill_keywords)
    doc = nlp(text)
    phrases = {chunk.text.strip().lower() for chunk in doc.noun_chunks if 2 < len(chunk.text.strip()) <= 40}
    return list(skill_keywords.union(phrases))

# Resume extraction with spaCy noun chunk fallback (filtered)
def extract_resume_skills(text, min_skills=5):
    skills = clean_skills(match_skills(text))
    if len(skills) < min_skills:
        doc = nlp(text)
        fallback_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if 2 < len(chunk.text.strip()) < 40]
        for chunk in fallback_chunks:
            for known_skill in skills:
                if known_skill.lower() in chunk.lower():
                    skills.append(known_skill)
                    break
    return list(set(skills))

def get_threshold(skill):
    return 0.55 if len(skill.split()) <= 2 else 0.65

def fuzzy_skill_match(jd_skills, resume_text):
    resume_skills = extract_resume_skills(resume_text)
    resume_skills = expand_synonyms(resume_skills)
    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)

    matched = set()
    unmatched = set()

    for skill in jd_skills:
        threshold = get_threshold(skill)
        skill_emb = model.encode(skill, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(skill_emb, resume_embeddings)[0]
        max_score = sims.max().item()
        if max_score >= threshold:
            matched.add(skill)
        else:
            unmatched.add(skill)

    return matched, unmatched

def compare_jd_resume(jd_text, resume_text, jd_embedding=None):
    # ✅ Avoid reprocessing JD embedding
    if jd_embedding is None:
        jd_embedding = model.encode(jd_text, convert_to_numpy=True)

    try:
        resume_embedding = model.encode(resume_text, convert_to_numpy=True)
    except RuntimeError:
        print("❌ GPU fallback retrying on CPU.")
        model._target_device = torch.device("cpu")
        resume_embedding = model.encode(resume_text, convert_to_numpy=True)

    sim_score = float(cosine_similarity([jd_embedding], [resume_embedding])[0][0])
    sim_score_normalized = round(sim_score * 10, 2)

    jd_skills = extract_jd_skills(jd_text)
    matched_fuzzy, missing_fuzzy = fuzzy_skill_match(jd_skills, resume_text)

    skill_score = round((len(matched_fuzzy) / max(1, len(jd_skills))) * 10, 2)
    final_score = round((sim_score_normalized * 0.2) + (skill_score * 0.8), 2)

    # Calculate skill match ratio: % of JD skills found in resume
    skill_match_ratio = len(matched_fuzzy) / max(1, len(jd_skills))

    # Final shortlist symbol based on ratio
    if skill_match_ratio >= 0.6:
        shortlist = "✅"
    elif 0.4 <= skill_match_ratio < 0.6:
        shortlist = "⚠️"
    else:
        shortlist = "❌"

    # Strict match only for visible UI
    resume_skills_clean = extract_resume_skills(resume_text)
    matched_strict = sorted(set(jd_skills) & set(resume_skills_clean))
    missing_strict = sorted(set(jd_skills) - set(matched_strict))

    return {
        "score": final_score,
        "jd_skills": sorted(jd_skills),
        "strengths": matched_strict,
        "gaps": missing_strict,
        "shortlist": shortlist,
        "mobile": extract_mobile(resume_text),
        "email": extract_email(resume_text)
    }
