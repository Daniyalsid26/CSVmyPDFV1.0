"""app.py — Gradio UI."""
import gradio as gr

from pipeline import process_pdfs

_CSS = """
body, .gradio-container {
    background: #f8fafc !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container { max-width: 700px !important; margin: 0 auto !important; }

.step-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px !important;
}
.step-label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .1em;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 10px;
    margin-bottom: 14px;
}

#hero { text-align: center; padding: 28px 0 4px; }
#hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin: 0 0 6px !important;
    line-height: 1.2 !important;
}
#tagline {
    text-align: center;
    color: #64748b !important;
    font-size: 1rem !important;
    margin-bottom: 24px !important;
}
#convert-btn {
    background: #0f172a !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px !important;
}
#status textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.8rem !important;
    color: #374151 !important;
    border-color: #e2e8f0 !important;
    background: #f8fafc !important;
}
"""

with gr.Blocks(css=_CSS, title="CSV my PDF") as demo:

    gr.HTML("""
        <div id="hero">
            <h1>CSV my PDF</h1>
        </div>
    """)
    gr.Markdown(
        "Turn bank statement PDFs into clean, structured spreadsheets — instantly.",
        elem_id="tagline",
    )

    with gr.Group(elem_classes="step-card"):
        gr.HTML('<div class="step-label">Step 1 &nbsp;&middot;&nbsp; Upload your PDF(s)</div>')
        pdf_input = gr.File(
            label="",
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath",
        )
        combine_checkbox = gr.Checkbox(
            label="Merge all uploaded files into a single CSV",
            value=False,
        )

    with gr.Group(elem_classes="step-card"):
        gr.HTML('<div class="step-label">Step 2 &nbsp;&middot;&nbsp; Convert</div>')
        convert_btn = gr.Button(
            "Convert to CSV",
            variant="primary",
            elem_id="convert-btn",
        )
        status_box = gr.Textbox(
            label="",
            placeholder="Status will appear here once conversion completes...",
            interactive=False,
            lines=2,
            elem_id="status",
        )

    with gr.Group(elem_classes="step-card"):
        gr.HTML('<div class="step-label">Step 3 &nbsp;&middot;&nbsp; Download</div>')
        csv_output = gr.File(label="", file_count="multiple")

    convert_btn.click(
        fn=process_pdfs,
        inputs=[pdf_input, combine_checkbox],
        outputs=[csv_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
