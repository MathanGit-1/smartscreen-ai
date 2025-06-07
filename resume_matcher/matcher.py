# ==================== Imports ====================
from sentence_transformers import SentenceTransformer, util
import torch
import re
import spacy

from resume_matcher.utils import extract_mobile, extract_email, clean_skills
from resume_matcher.skill_helpers import normalize_skill, apply_reverse_synonyms, expand_synonyms
from jd_parser.skill_matcher import match_skills
from config.skills import ROLE_BASED_SKILLS, SYNONYM_MAP  # Removed SKILL_SET


# ==================== Model Initialization ====================
nlp = spacy.load("en_core_web_sm")

try:
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    model.encode(["test"], convert_to_tensor=True)
    print("✅ Model loaded on: CUDA")
except Exception as e:
    print(f"⚠️ CUDA unavailable, falling back to CPU: {e}")
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    model.encode(["test"], convert_to_tensor=True)


# ==================== Resume Skill Extraction ====================
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


# ==================== Fuzzy Matching ====================
def fuzzy_skill_match(jd_skills, resume_text):
    resume_skills = extract_resume_skills(resume_text)
    resume_skills = expand_synonyms(resume_skills)
    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)

    matched = set()
    unmatched = set()

    print("\n====== Fuzzy Skill Matching Debug ======")
    for skill in jd_skills:
        threshold = get_threshold(skill)
        skill_emb = model.encode(skill, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(skill_emb, resume_embeddings)[0]

        best_idx = torch.argmax(sims).item()
        best_score = sims[best_idx].item()
        best_match = resume_skills[best_idx]

        print(f"🔍 JD Skill: {skill}")
        print(f"    🔗 Closest Resume Term: {best_match}")
        print(f"    📈 Similarity Score: {best_score:.3f} (Threshold: {threshold})")

        if best_score >= threshold:
            matched.add(skill)
        else:
            unmatched.add(skill)

    print(f"\nFuzzy Match - Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    print("Matched Skills:", sorted(matched))
    print("Unmatched Skills:", sorted(unmatched))
    print("=========================================\n")

    return matched, unmatched


# ==================== Final Matcher Entry Point ====================
def compare_jd_resume(jd_text, resume_text):
    # Step 1: Extract raw skills from JD
    jd_skills_raw = match_skills(jd_text)
    print("\n🔍 JD Raw Extracted Skills (before clean):", jd_skills_raw)

    # Step 2: Clean the raw skills (e.g., title case, trim, dedupe)
    jd_skills_raw = clean_skills(jd_skills_raw)
    print("🧼 JD Cleaned Skills (after clean_skills):", jd_skills_raw)

    # Step 3: Dynamically construct the full valid skill set from ROLE_BASED_SKILLS and SYNONYM_MAP
    all_valid_skills = {
        normalize_skill(skill)
        for skills in ROLE_BASED_SKILLS.values()
        for skill in skills
    } | set(SYNONYM_MAP.keys())

    print("📘 Total Valid Skills (normalized, sample):", sorted(all_valid_skills)[:30], "...")

    # Step 4: Try to filter JD skills using the normalized list
    jd_skills_filtered = []
    for s in jd_skills_raw:
        norm = normalize_skill(s)
        if norm in all_valid_skills:
            jd_skills_filtered.append(s)
        else:
            print(f"⚠️ Skill filtered out: '{s}' → normalized as '{norm}' (not in valid skill set)")

    # Step 5: If not enough skills left after filtering, fallback to raw
    if len(jd_skills_filtered) < 3:
        jd_skills = apply_reverse_synonyms(jd_skills_raw)
        jd_quality_flag = "⚠️ Fallback: JD had too few valid skills"
    else:
        jd_skills = apply_reverse_synonyms(jd_skills_filtered)
        jd_quality_flag = "✅ JD filtered cleanly"

    # Step 6: Compare JD and resume using fuzzy skill matching
    matched_skills, missing_skills = fuzzy_skill_match(jd_skills, resume_text)
    match_ratio = len(matched_skills) / max(1, len(jd_skills))
    match_percent = round(match_ratio * 100)
    match_summary = f"{len(matched_skills)} / {len(jd_skills)} matched ({match_percent}%)"

    # Step 7: Decide shortlist label based on match %
    shortlist = (
        "✅ Good Match" if match_ratio >= 0.6 else
        ("✳️ Partial Match" if match_ratio >= 0.4 else "⚠️Low match")
    )

    # Step 8: Semantic similarity score between entire JD and resume
    sim_score = util.pytorch_cos_sim(
        model.encode(jd_text, convert_to_tensor=True),
        model.encode(resume_text, convert_to_tensor=True)
    ).item()
    sim_score_normalized = round(sim_score * 10, 2)

    # Final debug info
    print("\n====== DEBUG INFO ======")
    print("🧠 Final JD Extracted Skills:", jd_skills)
    print("✅ Matched Skills:", matched_skills)
    print("❌ Gaps:", missing_skills)
    print("📊 Match Ratio:", match_summary)
    print("📈 Cosine Similarity Score:", sim_score_normalized)
    print("📎 JD Skill Filtering Status:", jd_quality_flag)
    print("=========================\n")

    return {
        "jd_skills": sorted(jd_skills),
        "strengths": sorted(matched_skills),
        "gaps": sorted(missing_skills),
        "match_summary": match_summary,
        "shortlist": shortlist,
        "mobile": extract_mobile(resume_text),
        "email": extract_email(resume_text),
        "semantic_score": sim_score_normalized
    }

