# ========== Standard Library ==========
import os
import time
from io import BytesIO
from datetime import datetime
import concurrent.futures

# ========== Third-Party Libraries ==========
import gradio as gr
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

# ========== Local Modules ==========
from jd_parser.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from jd_parser.field_extractor import extract_fields_from_text
from jd_parser.skill_matcher import match_skills
from resume_matcher.matcher import compare_jd_resume as ai_compare_jd_resume

# ========== Environment Setup ==========
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"\n===== SmartScreen.AI Launched at {datetime.now()} =====\n")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
print(f"✅ Model loaded on: {model.device}")
model.encode(["SmartScreen.AI Warm-up"], convert_to_tensor=True)

# ========== Global State ==========
current_data = []

# ========== Core Functions ==========
def clean_skills(raw_skills):
    return sorted(set(s.strip().title() for s in raw_skills))

def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    try:
        raw_bytes = file.read() if not os.path.exists(file.name) else open(file.name, "rb").read()
        if not raw_bytes:
            return "❌ Failed to read file: File stream is empty."
        if ext == ".pdf":
            return extract_text_from_pdf(BytesIO(raw_bytes))
        elif ext == ".docx":
            return extract_text_from_docx(BytesIO(raw_bytes))
        elif ext == ".txt":
            return extract_text_from_txt(BytesIO(raw_bytes))
        else:
            return "❌ Unsupported file type"
    except Exception as e:
        return f"❌ Failed to read file: {str(e)}"

def process_jd(input_mode, file, text_input):
    if input_mode == "Upload File" and file:
        text = extract_text(file)
        source = os.path.basename(file.name)
    elif input_mode == "Paste Text" and text_input.strip():
        text = text_input.strip()
        source = "Manual Input"
    else:
        return "❌ Please provide a JD input", ""
    if text.startswith("❌"):
        return "❌ Error reading JD file", text
    fields = extract_fields_from_text(text)
    skills = clean_skills(match_skills(text))
    summary = f"""
**JD ID**: {fields['jd_id']}  
**Role**: {fields['role'] or 'Not available'}  
**Years of Experience**: {fields['yoe'] or 'Not available'}  
**Notice Period**: {fields['notice_period'] or 'Not available'}  
**No. of Positions**: {fields['num_positions'] or 'Not available'}  
**Work Location**: {fields['work_location'] or 'Not available'}  
**Shift Timing**: {fields['shift_timing'] or 'Not available'}  

---

**Key Skills**: {', '.join(skills) or 'Not available'}
"""
    title = f"📌 JD Summary: {source} | Role: {fields['role'] or 'N/A'}"
    return title, summary

def compare_jd_multiple_resumes(jd_file, resume_files):
    global current_data
    if not jd_file or not resume_files:
        return [["❌ JD or Resumes missing", "", "", "", "", ""]], ""

    jd_text = extract_text(jd_file)
    if jd_text.startswith("❌"):
        return [[jd_text, "", "", "", "", ""]], ""

    jd_embedding = model.encode(jd_text, convert_to_numpy=True)
    resume_files = resume_files if isinstance(resume_files, list) else [resume_files]

    def process_resume(resume_file):
        resume_text = extract_text(resume_file)
        if resume_text.startswith("❌"):
            return [os.path.basename(resume_file.name), "❌ Error", "", 0, resume_text, "🔴 Reject"]
        result = ai_compare_jd_resume(jd_text, resume_text, jd_embedding=jd_embedding)
        score = result["score"]
        label = "🟢 Good Match" if score >= 5 else "🟠 Review" if score > 3 else "🔴 Reject"
        return [
            os.path.basename(resume_file.name),
            result["mobile"],
            result["email"],
            score,
            ", ".join(result["strengths"]),
            label
        ]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_resume, resume_files))
    elapsed = time.time() - start

    # 🧪 Final message
    message = f"✅ Ranked {len(results)} resumes in {elapsed:.2f} seconds"
    print(message)

    current_data = sorted(results, key=lambda x: x[3], reverse=True)
    return current_data, message

# ========== Gradio UI ==========
with gr.Blocks(title="SmartScreen.AI") as main_app:
    with gr.Group(visible=True) as login_ui:
        access_code = gr.Textbox(label="🔐 Enter Access Code", type="password")
        login_btn = gr.Button("Login")
        login_error = gr.Markdown(visible=False)

    with gr.Group(visible=False) as main_ui:
        with gr.Tabs():
            with gr.TabItem("Resume Match Scores for Selected JD"):
                gr.Markdown("### 🧠 AI-powered Resume Ranking")

                jd_file = gr.File(label="📁 Upload JD", file_types=[".pdf", ".docx", ".txt"])
                resume_files = gr.File(label="📄 Upload Resumes", file_types=[".pdf", ".docx", ".txt"], file_count="multiple")
                compare_btn = gr.Button("🔍 Compare and Rank", variant="primary")
                result_grid = gr.Dataframe(headers=["Resume", "Mobile", "Email", "Score (/10)", "Matching Skills", "Match Recommendation"], row_count=3)
                status_message = gr.Markdown()

                compare_btn.click(fn=compare_jd_multiple_resumes, inputs=[jd_file, resume_files], outputs=[result_grid, status_message])

                jd_file.change(fn=lambda _: [], inputs=jd_file, outputs=result_grid)
                resume_files.change(fn=lambda _: [], inputs=resume_files, outputs=result_grid)

                gr.Markdown("<hr>")

            with gr.TabItem("📂 JD Parser"):
                input_mode = gr.Radio(["Upload File", "Paste Text"], label="Select JD Input Mode", value="Upload File")
                file_input = gr.File(file_types=[".pdf", ".docx", ".txt"], visible=True, label="📁 Upload JD")
                text_input = gr.Textbox(lines=10, label="📜 Paste JD Content", visible=False)
                title_out = gr.Markdown()
                summary_out = gr.Markdown()
                submit_btn = gr.Button("Submit")

                input_mode.change(fn=lambda m: (gr.update(visible=m == "Upload File"), gr.update(visible=m == "Paste Text")),
                                  inputs=input_mode, outputs=[file_input, text_input])
                submit_btn.click(fn=process_jd, inputs=[input_mode, file_input, text_input], outputs=[title_out, summary_out])

    login_btn.click(fn=lambda code: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
                    if code == "1234" else (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value="❌ Invalid code.")),
                    inputs=access_code, outputs=[main_ui, login_ui, login_error])

# ========== Final Launch ==========
main_app.launch()
