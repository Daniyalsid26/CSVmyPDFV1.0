"""app.py — Gradio UI."""
import logging
import os
from typing import Optional

import gradio as gr

from pipeline import process_pdfs

logging.basicConfig(level=logging.INFO, format="%(message)s")

_logger = logging.getLogger("csvpdf.app")

_MAX_DOWNLOADS = 5  # pre-created download buttons (covers up to 5 separate CSVs)
_MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB per file


def _validate_pdf_paths(pdf_paths) -> Optional[str]:
    if not pdf_paths:
        return "Please upload at least one PDF."

    for path in pdf_paths:
        name = os.path.basename(path)
        if not name.lower().endswith(".pdf"):
            _logger.warning("Rejected upload %s: invalid file extension", name)
            return f"Invalid file type: {name}. Please upload .pdf files only."
        try:
            size = os.path.getsize(path)
        except OSError:
            _logger.warning("Rejected upload %s: could not read file size", name)
            return f"Could not read file: {name}."
        if size > _MAX_PDF_BYTES:
            _logger.warning("Rejected upload %s: file too large (%s bytes)", name, size)
            return f"File too large: {name}. Max size is 20 MB."
        try:
            with open(path, "rb") as f:
                header = f.read(4)
        except OSError:
            _logger.warning("Rejected upload %s: could not open file", name)
            return f"Could not open file: {name}."
        if header != b"%PDF":
            _logger.warning("Rejected upload %s: invalid PDF magic bytes", name)
            return f"Invalid PDF content: {name}."

    return None

_CSS = """
footer { visibility: hidden; }

.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}

#title { text-align: center; padding-top: 24px; }
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
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}

/* Status log — force dark readable text regardless of theme */
#status textarea {
    font-family: 'ui-monospace', 'Menlo', monospace !important;
    font-size: 0.78rem !important;
    background: #f8fafc !important;
    color: #111827 !important;
    border-radius: 6px !important;
    resize: none !important;
    padding: 8px 10px !important;
    border: 1px solid #e5e7eb !important;
}

/* Preview table — light grey bg, white text for readability */
#preview-table table {
    font-size: 0.8rem !important;
    background: #23272e !important;
    color: #fff !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
#preview-table thead th {
    background: #3a3f47 !important;
    font-weight: 600 !important;
    color: #fff !important;
}
#preview-table tbody td { color: #fff !important; }
#preview-table * { color: #fff !important; }

/* Download buttons — full-width, left-aligned label */
.dl-btn button {
    width: 100% !important;
    justify-content: flex-start !important;
    font-weight: 500 !important;
}
.zip-btn button {
    width: 100% !important;
    justify-content: center !important;
    font-weight: 700 !important;
    margin-top: 6px !important;
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

    # ── Step 2 — Convert ─────────────────────────────────────────────────────
    with gr.Column(variant="panel"):
        gr.Markdown("**Step 2 — Convert**")
        convert_btn = gr.Button("Convert to CSV", variant="primary", elem_id="convert-btn")
        status_box = gr.Textbox(
            show_label=False,
            placeholder="Conversion log will appear here…",
            interactive=False,
            lines=3,
            max_lines=8,
            elem_id="status",
        )

    # ── Transaction preview (always visible, fills in after conversion) ───────
    with gr.Column(variant="panel"):
        gr.Markdown("**Transaction Preview** (first 20 rows)")
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

        # Individual CSV download buttons — appear one by one as each file converts
        dl_btns = []
        for _ in range(_MAX_DOWNLOADS):
            dl_btns.append(
                gr.DownloadButton(
                    label="Download CSV",
                    visible=False,
                    variant="secondary",
                    elem_classes=["dl-btn"],
                )
            )

        # ZIP button — appears only when 2+ files are converted
        zip_btn = gr.DownloadButton(
            "⬇  Download All as ZIP",
            visible=False,
            variant="primary",
            elem_classes=["zip-btn"],
        )

    # ── Streaming event handler ───────────────────────────────────────────────
    async def _convert(pdf_paths, combine):
        validation_error = _validate_pdf_paths(pdf_paths)
        if validation_error:
            hidden_btns = [gr.update(visible=False) for _ in range(_MAX_DOWNLOADS)]
            yield (validation_error, [], *hidden_btns, gr.update(visible=False))
            return

        async for csv_paths, status, zip_path, preview in process_pdfs(
            pdf_paths, combine, False
        ):
            btn_updates = []
            for i in range(_MAX_DOWNLOADS):
                if i < len(csv_paths):
                    fname = os.path.basename(csv_paths[i])
                    btn_updates.append(
                        gr.update(value=csv_paths[i], label=f"⬇  Download  {fname}", visible=True)
                    )
                else:
                    btn_updates.append(gr.update(visible=False))

            zip_update = (
                gr.update(value=zip_path, label="⬇  Download All as ZIP", visible=True)
                if zip_path is not None
                else gr.update(visible=False)
            )

            yield (status, preview, *btn_updates, zip_update)

    convert_btn.click(
        fn=_convert,
        inputs=[pdf_input, combine_checkbox],
        outputs=[status_box, preview_table, *dl_btns, zip_btn],
    )

    gr.HTML(
        '<div id="powered-by">powered by <strong>Fin</strong>alt<span style="color:#ff4fa3;">O</span> AI</div>'
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/tmp"])

