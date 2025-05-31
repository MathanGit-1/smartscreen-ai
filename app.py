import gradio as gr
import os
from io import BytesIO

from jd_parser.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from jd_parser.field_extractor import extract_fields_from_text
from jd_parser.skill_matcher import match_skills
from resume_matcher.matcher import compare_jd_resume as ai_compare_jd_resume

def clean_skills(raw_skills):
    return sorted(set(s.strip().title() for s in raw_skills))

# ✅ Fully in-memory extractor with error handling
def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    try:
        if isinstance(file, str) or os.path.exists(file.name):
            with open(file.name, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = file.read()

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


# === JD Parser ===
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

# === JD vs Multiple Resume Matcher ===
def compare_jd_multiple_resumes(jd_file, resume_files):
    if not jd_file or not resume_files:
        return [["❌ JD or Resumes missing", "", "", "", ""]]

    jd_text = extract_text(jd_file)
    if jd_text.startswith("❌"):
        return [[jd_text, "", "", "", ""]]

    results = []
    resume_files = resume_files if isinstance(resume_files, list) else [resume_files]

    for resume_file in resume_files:
        resume_text = extract_text(resume_file)
        if resume_text.startswith("❌"):
            results.append([
                os.path.basename(resume_file.name),
                "❌ Error",
                0,
                resume_text,
                "No"
            ])
            continue

        result = ai_compare_jd_resume(jd_text, resume_text)
        results.append([
            os.path.basename(resume_file.name),
            result["mobile"],
            result["email"],
            result["score"],
            ", ".join(result["strengths"]),
            result["shortlist"]
        ])

    results = sorted(results, key=lambda x: x[3], reverse=True)
    return results

# === Gradio App ===
with gr.Blocks(title="SmartScreen.AI") as app:
    with gr.Group(visible=True) as login_ui:
        access_code = gr.Textbox(label="🔐 Enter Access Code", type="password")
        login_btn = gr.Button("Login")
        login_error = gr.Markdown(visible=False)

    with gr.Group(visible=False) as main_ui:
        with gr.Tabs():
            with gr.TabItem("Resume Match Scores for Selected JD"):
                gr.Markdown("""
                <h3 style='text-align: center; font-weight: 700; font-size: 20px; margin-bottom: 10px; color: white;'>
                🧠 AI-powered resume ranking — <i>helps you identify top matches</i>.
                </h3>
                """)
                jd_file = gr.File(label="📁 Upload JD", file_types=[".pdf", ".docx", ".txt"])
                resume_files = gr.File(label="📄 Upload Resumes", file_types=[".pdf", ".docx", ".txt"], file_count="multiple")
                compare_btn = gr.Button("Compare and Rank")
                result_grid = gr.Dataframe(headers=["Resume", "Mobile","Email", "Score (/10)", "Matching Skills", "Shortlist?"], row_count=3)
                compare_btn.click(compare_jd_multiple_resumes, inputs=[jd_file, resume_files], outputs=result_grid)
                # 👇 Add this line below the grid
                gr.Markdown("""
                <div style='background-color:#f0f0f0; padding:10px; border-radius:8px; text-align:center; font-weight:bold; color:#333; font-size:15px;'>
                🔐 Files are processed in-memory and never stored.
                </div>
                """)
                # Reset grid when JD or resumes are cleared
                jd_file.change(fn=lambda _: [], inputs=jd_file, outputs=result_grid)
                resume_files.change(fn=lambda _: [], inputs=resume_files, outputs=result_grid)

            with gr.TabItem("📂 JD Parser"):
                gr.Markdown("### Extract structured info and skills from a JD.\n🔐 Files are processed in-memory and never stored.")
                input_mode = gr.Radio(choices=["Upload File", "Paste Text"], label="Select JD Input Mode", value="Upload File")
                file_input = gr.File(file_types=[".pdf", ".docx", ".txt"], visible=True, label="📁 Upload JD")
                text_input = gr.Textbox(lines=10, label="📜 Paste JD Content", visible=False)
                title_out = gr.Markdown()
                summary_out = gr.Markdown()
                submit_btn = gr.Button("Submit")

                def toggle_inputs(mode):
                    return (gr.update(visible=True), gr.update(visible=False)) if mode == "Upload File" else (gr.update(visible=False), gr.update(visible=True))

                input_mode.change(fn=toggle_inputs, inputs=input_mode, outputs=[file_input, text_input])
                submit_btn.click(fn=process_jd, inputs=[input_mode, file_input, text_input], outputs=[title_out, summary_out])

    def validate(code):
        if code == "1357":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value="❌ Invalid code. Please try again.")

    login_btn.click(fn=validate, inputs=access_code, outputs=[main_ui, login_ui, login_error])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)