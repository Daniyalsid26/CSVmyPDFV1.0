"""app.py — Gradio UI."""
import gradio as gr

from pipeline import process_pdfs

_CSS = """
footer { visibility: hidden; }

.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
}

#title {
    text-align: center;
    padding-top: 24px;
}
#title h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}
#tagline p {
    text-align: center;
    color: #6b7280;
    font-size: 0.92rem !important;
    margin-top: 0 !important;
    margin-bottom: 28px !important;
}

#convert-btn {
    width: 100% !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-top: 8px !important;
}

#status textarea {
    font-family: 'ui-monospace', 'Menlo', monospace !important;
    font-size: 0.78rem !important;
    background: #f8fafc !important;
    border-radius: 6px !important;
    resize: none !important;
    padding: 8px 10px !important;
    border: 1px solid #e5e7eb !important;
}

#preview-table table {
    font-size: 0.8rem !important;
}
#preview-table thead th {
    background: #f1f5f9 !important;
    font-weight: 600 !important;
}

.download-hint p {
    color: #6b7280;
    font-size: 0.82rem !important;
    margin: 0 0 6px 0 !important;
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

    # ── Step 1 — Upload ──────────────────────────────────────────────────────
    with gr.Column(variant="panel"):
        gr.Markdown("**Step 1 — Upload your PDF(s)**")
        pdf_input = gr.File(
            show_label=False,
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath",
        )
        with gr.Row():
            combine_checkbox = gr.Checkbox(
                label="Merge all files into one CSV",
                value=False,
            )
            is_scanned_checkbox = gr.Checkbox(
                label="My statements have been scanned (Use OCR, slower but more accurate)",
                value=False,
            )

    # ── Step 2 — Convert ─────────────────────────────────────────────────────
    with gr.Column(variant="panel"):
        gr.Markdown("**Step 2 — Convert**")
        convert_btn = gr.Button(
            "⚡ Convert to CSV",
            variant="primary",
            elem_id="convert-btn",
        )
        status_box = gr.Textbox(
            show_label=False,
            placeholder="Conversion log will appear here…",
            interactive=False,
            lines=3,
            max_lines=8,
            elem_id="status",
        )

    # ── Transaction preview ───────────────────────────────────────────────────
    with gr.Accordion("Transaction Preview (first 20 rows)", open=False) as preview_accordion:
        preview_table = gr.Dataframe(
            headers=["Date", "Type", "Details", "Paid Out", "Paid In", "Balance"],
            datatype=["str"] * 6,
            interactive=False,
            wrap=True,
            elem_id="preview-table",
        )

    # ── Step 3 — Download ────────────────────────────────────────────────────
    with gr.Column(variant="panel"):
        gr.Markdown("**Step 3 — Download**")
        gr.Markdown("Click a filename below to download.", elem_classes=["download-hint"])
        csv_output = gr.File(
            show_label=False,
            file_count="multiple",
            interactive=False,
        )
        zip_output = gr.File(
            label="⬇ Download All as ZIP",
            file_count="single",
            interactive=False,
            visible=False,
        )

    # ── Streaming event handler ───────────────────────────────────────────────
    async def _convert(pdf_paths, combine, is_scanned):
        async for csv_paths, status, zip_path, preview in process_pdfs(
            pdf_paths, combine, is_scanned
        ):
            yield (
                csv_paths,
                status,
                gr.update(value=zip_path, visible=zip_path is not None),
                preview,
                gr.update(open=bool(preview)),
            )

    convert_btn.click(
        fn=_convert,
        inputs=[pdf_input, combine_checkbox, is_scanned_checkbox],
        outputs=[csv_output, status_box, zip_output, preview_table, preview_accordion],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/tmp"])

