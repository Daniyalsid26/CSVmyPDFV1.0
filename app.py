"""app.py — Gradio UI."""
import gradio as gr

from pipeline import process_pdfs

_CSS = """
footer { visibility: hidden; }

.gradio-container {
    max-width: 680px !important;
    margin: 0 auto !important;
}

#title {
    text-align: center;
    padding-top: 24px;
}
#title h1 {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}
#tagline p {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem !important;
    margin-top: 0 !important;
    margin-bottom: 28px !important;
}

#upload-col { padding-bottom: 0 !important; }

#convert-btn {
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

#status textarea {
    font-family: 'ui-monospace', 'Menlo', monospace !important;
    font-size: 0.78rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    resize: none !important;
    padding: 4px 0 !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate"),
    css=_CSS,
    title="CSV my PDF",
) as demo:

    gr.Markdown("# CSV my PDF", elem_id="title")
    gr.Markdown(
        "Turn bank statement PDFs into clean, structured spreadsheets — instantly.",
        elem_id="tagline",
    )

    with gr.Column(variant="panel", elem_id="upload-col"):
        gr.Markdown("**1. Upload your PDF(s)**")
        pdf_input = gr.File(
            show_label=False,
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath",
        )
        combine_checkbox = gr.Checkbox(
            label="Merge all files into one CSV",
            value=False,
        )
        is_scanned_checkbox = gr.Checkbox(
            label="These are scanned PDFs (use OCR extraction)",
            value=False,
        )

    with gr.Column(variant="panel"):
        gr.Markdown("**2. Convert**")
        convert_btn = gr.Button(
            "Convert to CSV",
            variant="primary",
            elem_id="convert-btn",
        )
        status_box = gr.Textbox(
            show_label=False,
            placeholder="Conversion status will appear here…",
            interactive=False,
            lines=2,
            elem_id="status",
        )

    with gr.Column(variant="panel"):
        gr.Markdown("**3. Download**")
        csv_output = gr.File(show_label=False, file_count="multiple")

    convert_btn.click(
        fn=process_pdfs,
        inputs=[pdf_input, combine_checkbox, is_scanned_checkbox],
        outputs=[csv_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
