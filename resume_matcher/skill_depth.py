import re
import spacy
from config.skills import SYNONYM_MAP, ACTION_VERBS, EXPERIENCE_HEADERS

# Load spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Ensure ACTION_VERBS is a set
ACTION_VERBS = set(ACTION_VERBS)


def extract_experience_sections(text):
    """
    Extracts only the 'Experience', 'Projects', etc. sections (as defined in EXPERIENCE_HEADERS).
    Keeps paragraphs even across blank lines unless an unrelated header is hit.
    """
    lines = text.splitlines()
    sections = []
    buffer = []
    in_exp = False

    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if any(h in line_lower for h in EXPERIENCE_HEADERS):
            in_exp = True
            if buffer:
                sections.append(" ".join(buffer))
                buffer = []
            continue

        if any(h in line_lower for h in ["technical skills", "skills", "summary", "tools"]):
            in_exp = False

        if in_exp and line_clean:
            buffer.append(line_clean)

    if buffer:
        sections.append(" ".join(buffer))

    print(f"📌 Extracted Experience Sections: {len(sections)}")
    return sections if sections else [text]  # fallback to full resume


def evaluate_skill_depth(resume_text, matched_skills):
    """
    Categorizes each skill into:
    🛠️ Strong Mention: Found in experience with action verb
    📌 Weak Mention: Found without action context or outside projects
    ◾️ No Mention: Not found in resume

    Additionally returns justification:
    {
        skill: {
            'tag': 🛠️ or 📌 or ◾️,
            'source': "experience" / "resume",
            'trigger': synonym used,
            'sentence': actual sentence where it appeared
        }
    }
    """
    skill_scores = {}
    sections = extract_experience_sections(resume_text)
    resume_lower = resume_text.lower()

    for skill in matched_skills:
        skill_lower = skill.lower()
        synonyms = SYNONYM_MAP.get(skill_lower, [skill_lower])
        justification = {
            "tag": "◾️ No Mention",
            "source": "",
            "trigger": "",
            "sentence": ""
        }

        # 1. Inside Experience Section
        for section in sections:
            doc = nlp(section)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                sent_lower = sent_text.lower()
                for syn in synonyms:
                    if syn in sent_lower:
                        if any(verb in sent_lower for verb in ACTION_VERBS):
                            justification.update({
                                "tag": "🛠️ Strong Mention",
                                "source": "experience",
                                "trigger": syn,
                                "sentence": sent_text
                            })
                            break
                        elif justification["tag"] == "◾️ No Mention":
                            justification.update({
                                "tag": "📌 Weak Mention",
                                "source": "experience",
                                "trigger": syn,
                                "sentence": sent_text
                            })
            if justification["tag"] == "🛠️ Strong Mention":
                break  # no need to keep searching

        # 2. Fallback to entire resume if not yet found
        if justification["tag"] == "◾️ No Mention":
            for syn in synonyms:
                if syn in resume_lower:
                    justification.update({
                        "tag": "📌 Weak Mention",
                        "source": "resume",
                        "trigger": syn,
                        "sentence": ""
                    })
                    break

        skill_scores[skill] = justification

    return skill_scores
