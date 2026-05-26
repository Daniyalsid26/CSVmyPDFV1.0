"""app.py — Gradio UI. All pipeline logic lives in pipeline.py."""
import gradio as gr

from pipeline import process_pdf


_CSS = """
body, .gradio-container { font-family: 'Courier New', monospace !important; }
#title { text-align: center; }
"""

with gr.Blocks(css=_CSS, title="CSV my PDF") as demo:
    gr.Markdown("# CSV my PDF", elem_id="title")
    gr.Markdown(
        "Upload a bank statement PDF (digital or scanned). "
        "The AI extracts every transaction and returns a clean CSV."
    )

    pdf_input = gr.File(
        label="Bank Statement (PDF)",
        file_types=[".pdf"],
        type="filepath",
    )
    convert_btn = gr.Button("Convert to CSV", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False, lines=1)
    csv_output = gr.File(label="Download CSV")

    convert_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[csv_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
