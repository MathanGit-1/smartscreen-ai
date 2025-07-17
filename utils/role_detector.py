import re
from config.skills import ROLE_SYNONYMS

def auto_detect_role(jd_text: str) -> str:
    print("🔍 Running auto role detection...")
    jd_lower = jd_text.lower()

    for role, keywords in ROLE_SYNONYMS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, jd_lower):
                print(f"✅ Matched via synonym: {keyword} → Role: {role}")
                return role

    print("⚠️ No match via ROLE_SYNONYMS. Returning 'Others'")
    return "Others"
