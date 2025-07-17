import os
import time
from io import BytesIO
from sentence_transformers import util

from jd_parser.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from jd_parser.skill_matcher import match_skills
from resume_matcher.matcher import compare_jd_resume
from config.skills import ROLE_BASED_SKILLS, SYNONYM_MAP

# ========= Normalize Skill =========
def normalize(skill):
    for canonical, variants in SYNONYM_MAP.items():
        if skill.lower().strip() in [v.lower().strip() for v in variants]:
            return canonical.lower().strip()
    return skill.lower().strip()

# ========= Get Role Score Mapping =========
def get_role_scores(text):
    raw_skills = match_skills(text)
    flattened = [item for sublist in raw_skills for item in (sublist if isinstance(sublist, list) else [sublist])]
    normalized = set([normalize(skill) for skill in flattened])
    role_scores = {}

    for role, keywords in ROLE_BASED_SKILLS.items():
        normalized_keywords = set([normalize(k) for k in keywords])
        matched = normalized & normalized_keywords
        score = len(matched) / len(normalized_keywords) if normalized_keywords else 0
        role_scores[role] = score

    return role_scores

# ========= Infer Top Role =========
def infer_resume_role(text):
    role_scores = get_role_scores(text)
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_roles[0][0] if sorted_roles and sorted_roles[0][1] >= 0.15 else "unknown"

# ========= File Reader =========
def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    try:
        if hasattr(file, "read"):
            content = file.read()
        elif hasattr(file, "name") and os.path.isfile(file.name):
            with open(file.name, "rb") as f:
                content = f.read()
        else:
            return None, "❌ Could not read file content."

        if not content:
            return None, "❌ File is empty."

        if ext == ".pdf":
            return extract_text_from_pdf(BytesIO(content)), None
        elif ext == ".docx":
            return extract_text_from_docx(BytesIO(content)), None
        elif ext == ".txt":
            return extract_text_from_txt(BytesIO(content)), None
        else:
            return None, "❌ Unsupported file format."
    except Exception as e:
        return None, f"❌ Extraction failed: {str(e)}"

# ========= Main Comparison =========
def compare_multiple_jds_resumes(jd_files, resume_files):
    print('inside compare_multiple_jds_resumes')
    if not jd_files or not resume_files:
        return "<b>❌ Please upload both JD and Resume files.</b>", ""

    start = time.time()
    html_blocks = []

    for jd_file in jd_files:
        jd_text, error = extract_text(jd_file)
        jd_name = os.path.basename(jd_file.name)

        if error:
            html_blocks.append(f"<h3>{jd_name}</h3><p>{error}</p>")
            continue

        jd_role = infer_resume_role(jd_text)

        jd_block = f"""
<details style='margin-bottom:15px; border:1px solid #444; border-radius:8px; background-color:white; color:white; padding:10px;'>
  <summary style='font-weight:bold; font-size:18px; color:#FF6600;'>{jd_name} <span style='color:gray;'>({jd_role})</span></summary>
  <div style='padding:10px;'>
    <table style='width:100%; border-collapse: collapse; font-size:14px;'>
      <thead>
        <tr style='background-color:#FF6600; color:white; font-weight:bold;'>
          <th style='padding:10px; border:1px solid #333;'>Resume</th>
          <th style='padding:10px; border:1px solid #333;'>Mobile</th>
          <th style='padding:10px; border:1px solid #333;'>Match %</th>
          <th style='padding:10px; border:1px solid #333;'>Shortlist</th>
          <th style='padding:10px; border:1px solid #333;'>JD Skills Matched</th>
          <th style='padding:10px; border:1px solid #333;'>Gaps</th>
        </tr>
      </thead>
      <tbody>
"""

        resume_rows = []

        for resume_file in resume_files:
            resume_text, error = extract_text(resume_file)
            resume_name = os.path.basename(resume_file.name)

            if error:
                resume_rows.append((
                    0,
                    f"""
        <tr>
          <td style='padding:10px; border:1px solid #333;'>{resume_name}</td>
          <td colspan='5' style='padding:10px; border:1px solid #333;'>{error}</td>
        </tr>
"""
                ))
                continue

            resume_role = infer_resume_role(resume_text)
            if jd_role.lower() not in resume_text.lower():
                continue

            result = compare_jd_resume(jd_text, resume_text)

            skill_html = ""
            for skill in result["jd_skills"]:
                tag = result["skill_justification"].get(skill, {}).get("tag", "")
                trigger = result["skill_justification"].get(skill, {}).get("trigger", "")
                if tag == "🛠️ Strong Mention":
                    skill_html += f"<span title='Matched via: {trigger}' style='margin-right:6px;'>🛠️ {skill}</span>"
                elif tag == "📌 Weak Mention":
                    skill_html += f"<span title='Matched via: {trigger}' style='margin-right:6px;'>📌 {skill}</span>"

            gap_html = ", ".join(result["gaps"])

            resume_rows.append((
                result.get("score", 0),
                f"""
        <tr>
          <td style='padding:10px; border:1px solid #333; color:black;'>{resume_name}</td>
          <td style='padding:10px; border:1px solid #333; color:black;'>{result["mobile"] or "Not found"}</td>
          <td style='padding:10px; border:1px solid #333; color:black;'>{result["match_summary"]}</td>
          <td style='padding:10px; border:1px solid #333; color:black;'>{result["shortlist"]}</td>
          <td style='padding:10px; border:1px solid #333; color:black;'>{skill_html}</td>
          <td style='padding:10px; border:1px solid #333; color:black;'>{gap_html}</td>
        </tr>
"""
            ))

        resume_rows.sort(key=lambda x: x[0], reverse=True)

        for _, row_html in resume_rows:
            jd_block += row_html

        jd_block += """
      </tbody>
    </table>
  </div>
</details>
"""
        html_blocks.append(jd_block)

    full_html = "<div style='padding: 10px;'>" + "".join(html_blocks) + "</div>"
    elapsed = time.time() - start
    status_msg = f"✅ Ranked {len(resume_files)} resumes in {elapsed:.2f} seconds"

    return full_html, status_msg