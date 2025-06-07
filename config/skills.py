import json
import os

# Load the full skills.json file
skills_path = os.path.join(os.path.dirname(__file__), "skills.json")

with open(skills_path, "r", encoding="utf-8") as f:
    skills_data = json.load(f)

# ✅ These two objects will be imported across the app
ROLE_BASED_SKILLS = skills_data["skills_by_role"]
SYNONYM_MAP = skills_data["synonyms"]
ROLE_SYNONYMS = skills_data["role_synonyms"]
SPECIAL_CHARACTER_SKILLS =  [s.lower() for s in skills_data["special_character_skills"]]

# Build flat master skill list from role-based mapping
flat_skills = set()
for role_skills in skills_data.get("skills_by_role", {}).values():
    flat_skills.update(role_skills)

skills_data["skills"] = sorted(flat_skills)
SKILL_SET = skills_data