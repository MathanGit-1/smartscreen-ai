from sentence_transformers import SentenceTransformer, util
import torch
import spacy

from resume_matcher.utils import extract_mobile, extract_email, clean_skills
from resume_matcher.skill_helpers import normalize_skill, apply_reverse_synonyms, expand_synonyms
from jd_parser.skill_matcher import match_skills
from resume_matcher.skill_depth import evaluate_skill_depth
from config.skills import ROLE_BASED_SKILLS, SYNONYM_MAP

# ========== Model & NLP Init ==========
nlp = spacy.load("en_core_web_sm")

try:
    model_jobbert = SentenceTransformer("TechWolf/JobBERT-v2", device="cpu")
    model_jobbert.encode(["test"], convert_to_tensor=True)
    #print("✅ JobBERT model loaded successfully")
except Exception as e:
    print(f"❌ JobBERT loading failed: {e}")
    model_jobbert = None

# ========== Helpers ==========
def extract_resume_skills(text, skill_list=None, min_skills=5):
    skills = clean_skills(match_skills(text, skill_list=skill_list))
    if len(skills) < min_skills:
        doc = nlp(text)
        fallback = [
            chunk.text.strip()
            for chunk in doc.noun_chunks
            if 2 < len(chunk.text.strip()) < 40
        ]
        skills += fallback
    return apply_reverse_synonyms(set(skills))

def get_threshold(skill):
    return 0.55 if len(skill.split()) <= 2 else 0.65

# ========== Skill Matcher ==========
def fuzzy_skill_match(jd_skills, resume_text):
    resume_skills = extract_resume_skills(resume_text)
    resume_skills = expand_synonyms(resume_skills)
    resume_embeddings = model_jobbert.encode(resume_skills, convert_to_tensor=True)

    matched = set()
    unmatched = set()
    match_sources = {}

    for skill in jd_skills:
        threshold = get_threshold(skill)
        skill_emb = model_jobbert.encode(skill, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(skill_emb, resume_embeddings)[0]

        best_idx = torch.argmax(sims).item()
        best_score = sims[best_idx].item()
        best_match = resume_skills[best_idx]

        if best_score >= threshold:
            matched.add(skill)
            match_sources[skill] = best_match
        else:
            unmatched.add(skill)

    return matched, unmatched, match_sources

# ========== Main Function ==========
def compare_jd_resume(jd_text, resume_text):
    jd_skills_raw = match_skills(jd_text)
    jd_skills_raw = clean_skills(jd_skills_raw)

    all_valid_skills = {
        normalize_skill(skill)
        for skills in ROLE_BASED_SKILLS.values()
        for skill in skills
    } | set(SYNONYM_MAP.keys())

    jd_skills_filtered = [s for s in jd_skills_raw if normalize_skill(s) in all_valid_skills]
    jd_skills = apply_reverse_synonyms(jd_skills_filtered) if len(jd_skills_filtered) >= 3 else apply_reverse_synonyms(jd_skills_raw)
    #print(f"📌 Extracted JD Skills: {jd_skills}")

    matched_skills, missing_skills, match_sources = fuzzy_skill_match(jd_skills, resume_text)

    skill_depth = evaluate_skill_depth(resume_text, jd_skills)
    #print(f"🔍 Skill Justification (raw): {skill_depth}")

    tooltip_justification = {}
    for skill in jd_skills:
        tag = skill_depth.get(skill, {}).get("tag", "◾️ No Mention")
        trigger = match_sources.get(skill)
        if trigger and tag != "◾️ No Mention":
            tooltip_justification[skill] = {
                "tag": tag,
                "source": skill_depth[skill].get("source", ""),
                "trigger": trigger,
                "sentence": skill_depth[skill].get("sentence", "")
            }
        else:
            tooltip_justification[skill] = {
                "tag": tag,
                "source": "",
                "trigger": "",
                "sentence": ""
            }

    # Weighted Scoring
    weights = {
        "🛠️ Strong Mention": 1.0,
        "📌 Weak Mention": 0.5,
        "◾️ No Mention": 0.0
    }

    strong_count = 0
    weak_count = 0
    weighted_score = 0.0

    for skill in jd_skills:
        tag = skill_depth.get(skill, {}).get("tag", "◾️ No Mention")
        if tag == "🛠️ Strong Mention":
            strong_count += 1
            weighted_score += 1.0
        elif tag == "📌 Weak Mention":
            weak_count += 1
            weighted_score += 0.5

    total = max(1, len(jd_skills))
    weighted_percent = round((weighted_score / total) * 100)

    # Match display in format: 75% weighted (🛠️+📌 = 3.0 / 4)
    match_summary = f"{weighted_percent}% weighted (🛠️+📌 = {weighted_score:.1f} / {total})"

    shortlist = (
        "✅ Good Match" if weighted_percent >= 60 else
        ("✳️ Partial Match" if weighted_percent >= 40 else "⚠️ Low match")
    )

    return {
        "jd_skills": sorted(jd_skills),
        "strengths": sorted([s for s in jd_skills if skill_depth.get(s, {}).get("tag") in ["🛠️ Strong Mention", "📌 Weak Mention"]]),
        "gaps": sorted([s for s in jd_skills if skill_depth.get(s, {}).get("tag") == "◾️ No Mention"]),
        "match_summary": match_summary,
        "shortlist": shortlist,
        "mobile": extract_mobile(resume_text),
        "email": extract_email(resume_text),
        "skill_justification": tooltip_justification,
        "weighted_score": weighted_score,
        "total_skills": total,
        "strong_count": strong_count,
        "weak_count": weak_count
    }
