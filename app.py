# app.py
import gradio as gr
import os
from jd_parser.extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from jd_parser.field_extractor import extract_fields_from_text
from jd_parser.skill_matcher import match_skills

def clean_skills(raw_skills):
    return sorted(set(s.strip().title() for s in raw_skills))

def process_jd(input_mode, file, text_input):
    if input_mode == "Upload File" and file:
        filename = file.name
        ext = os.path.splitext(filename)[-1].lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(filename)
        elif ext == ".docx":
            text = extract_text_from_docx(filename)
        elif ext == ".txt":
            text = extract_text_from_txt(file)
        else:
            return "❌ Unsupported file format", ""

        source = os.path.basename(filename)

    elif input_mode == "Paste Text" and text_input.strip():
        text = text_input.strip()
        source = "Manual Input"
    else:
        return "❌ Please provide a valid JD via file or pasted content", ""

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

# Gradio UI
with gr.Blocks(title="SmartScreen.AI - JD Analyzer") as app:
    gr.Markdown("### 📂 Upload a job description or paste content to extract structured data and suggest Naukri filters.")

    input_mode = gr.Radio(choices=["Upload File", "Paste Text"], label="Select JD Input Mode", value="Upload File")
    file_input = gr.File(file_types=[".pdf", ".docx", ".txt"], visible=True, label="📁 Upload JD")
    text_input = gr.Textbox(lines=10, label="📝 Paste JD Content", visible=False)

    title_out = gr.Markdown()
    summary_out = gr.Markdown()
    submit_btn = gr.Button("Submit")

    def toggle_inputs(mode):
        if mode == "Upload File":
            return gr.File(visible=True), gr.Textbox(visible=False, value="")
        else:
            return gr.File(visible=False), gr.Textbox(visible=True, value="")

    input_mode.change(
        fn=toggle_inputs,
        inputs=input_mode,
        outputs=[file_input, text_input]
    )

    submit_btn.click(fn=process_jd, inputs=[input_mode, file_input, text_input], outputs=[title_out, summary_out])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
