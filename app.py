# ========== Standard Library ==========
import os
import time
from io import BytesIO
from datetime import datetime
import concurrent.futures
import tempfile

# ========== Third-Party Libraries ==========
import gradio as gr
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

# ========== Local Modules ==========
from jd_parser.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from jd_parser.field_extractor import extract_fields_from_text
from jd_parser.skill_matcher import match_skills
from resume_matcher.matcher import compare_jd_resume

# ========== Environment Setup ==========
print(f"\n===== SmartScreen.AI Launched at {datetime.now()} =====\n")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
model.encode(["SmartScreen.AI Warm-up"], convert_to_tensor=True)
print(f"✅ Model loaded on: {model.device}")

# ========== Global State ==========
current_data = []
excel_ready = gr.State(value=False)

# ========== Utility Functions ==========
def clean_skills(raw_skills):
    return sorted(set(s.strip().title() for s in raw_skills))

def extract_text(file):
    print(f"📄 Reading file: {file.name}")
    ext = os.path.splitext(file.name)[-1].lower()
    try:
        raw_bytes = file.read() if not os.path.exists(file.name) else open(file.name, "rb").read()
        print(f"📦 Read {len(raw_bytes)} bytes from file")
        if not raw_bytes:
            return "❌ Failed to read file: File stream is empty."

        if ext == ".pdf":
            print("📥 Extracting PDF content...")
            text = extract_text_from_pdf(BytesIO(raw_bytes))
        elif ext == ".docx":
            print("📥 Extracting DOCX content...")
            text = extract_text_from_docx(BytesIO(raw_bytes))
        elif ext == ".txt":
            print("📥 Extracting TXT content...")
            text = extract_text_from_txt(BytesIO(raw_bytes))
        else:
            print("❌ Unsupported file type")
            return "❌ Unsupported file type"

        print(f"📃 Extracted text length: {len(text)} characters")
        return text

    except Exception as e:
        print(f"❌ Exception in extract_text: {str(e)}")
        return f"❌ Failed to read file: {str(e)}"

# ========== Main JD vs Resumes Matching ==========
def compare_jd_multiple_resumes(jd_file, resume_files):
    global current_data
    if not jd_file or not resume_files:
        return [["❌ JD or Resumes missing", "", "", "", "", ""]], ""

    jd_text = extract_text(jd_file)
    if jd_text.startswith("❌"):
        return [[jd_text, "", "", "", "", ""]], ""

    resume_files = resume_files if isinstance(resume_files, list) else [resume_files]

    def process_resume(resume_file):
        print(f"\n🧾 Processing resume: {resume_file.name}")
        resume_text = extract_text(resume_file)

        if resume_text.startswith("❌"):
            print(f"❌ Error in resume text: {resume_text}")
            return [os.path.basename(resume_file.name), "❌ Error", "", "", "", "🔴 Reject", 0]

        print("🧠 Comparing with JD...")
        result = compare_jd_resume(jd_text, resume_text)
        print(f"📊 Result from compare_jd_resume: {result}")

        ratio = result["match_summary"]

        try:
            percent_value = int(ratio.split("(")[-1].replace("%)", ""))
        except Exception as e:
            print(f"⚠️ Failed to parse percent from ratio: {ratio}, error: {str(e)}")
            percent_value = 0

        return [
            os.path.basename(resume_file.name),
            result["mobile"],
            ", ".join(result["strengths"]),
            ", ".join(result["gaps"]),
            ratio,
            result["shortlist"],
            percent_value
        ]


    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_resume, resume_files))
    elapsed = time.time() - start

    # ✅ Sort by match % (last column), then remove it before showing in grid
    sorted_results = sorted(results, key=lambda x: x[-1], reverse=True)
    current_data = [r[:-1] for r in sorted_results]

    print(f"✅ Ranked {len(current_data)} resumes in {elapsed:.2f} seconds")
    return current_data, f"✅ Ranked {len(current_data)} resumes in {elapsed:.2f} seconds"

# ========== Excel Export ==========
def generate_excel_download():
    if not current_data:
        print("⚠️ No data to export")
        return gr.update(value=None, visible=False)

    print(f"📊 Exporting {len(current_data)} rows to Excel...")
    df = pd.DataFrame(current_data, columns=[
        "Resume", "Mobile", "JD Skills Matched", "Gaps", "Match %", "Shortlist"
    ])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Top Matches")
        tmp_path = tmp.name

    print(f"✅ File ready at: {tmp_path}")
    return gr.update(value=tmp_path, visible=True)

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

                result_grid = gr.Dataframe(
                    headers=["Resume", "Mobile", "JD Skills Matched", "Gaps", "Match %", "Shortlist"],
                    row_count=3
                )
                status_message = gr.Markdown()

                generate_btn = gr.Button("📥 Prepare Excel for Download")
                download_btn = gr.DownloadButton(label="⬇️ Click to Download", visible=False)

                generate_btn.click(fn=generate_excel_download, inputs=[], outputs=[download_btn])
                compare_btn.click(fn=compare_jd_multiple_resumes, inputs=[jd_file, resume_files], outputs=[result_grid, status_message])
                download_btn.click(fn=generate_excel_download, inputs=[], outputs=[download_btn])

                jd_file.change(fn=lambda: gr.update(visible=False), inputs=[], outputs=[download_btn])
                resume_files.change(fn=lambda: gr.update(visible=False), inputs=[], outputs=[download_btn])

    login_btn.click(
        fn=lambda code: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
        if code == "1234"
        else (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value="❌ Invalid code.")),
        inputs=access_code,
        outputs=[main_ui, login_ui, login_error]
    )

# ========== Launch ==========
main_app.launch()
