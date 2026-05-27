"""app.py — Gradio UI."""
import os

import gradio as gr

from pipeline import process_pdfs

_MAX_DOWNLOADS = 5  # pre-created download buttons (covers up to 5 separate CSVs)

_CSS = """
footer { visibility: hidden; }

.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
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

/* Preview table — force dark readable text on all cells */
#preview-table table { font-size: 0.8rem !important; }
#preview-table thead th {
    background: #f1f5f9 !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
#preview-table tbody td { color: #111827 !important; }
#preview-table * { color: #111827 !important; }

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
            is_scanned_checkbox = gr.Checkbox(
                label="My statements have been scanned (Use OCR, slower but more accurate)",
                value=False,
            )

    # ── Step 2 — Convert ─────────────────────────────────────────────────────
    with gr.Column(variant="panel"):
        gr.Markdown("**Step 2 — Convert**")
            convert_btn = gr.Button("Convert to CSV", variant="primary", elem_id="convert-btn")
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
    async def _convert(pdf_paths, combine, is_scanned):
        async for csv_paths, status, zip_path, preview in process_pdfs(
            pdf_paths, combine, is_scanned
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
        inputs=[pdf_input, combine_checkbox, is_scanned_checkbox],
        outputs=[status_box, preview_table, *dl_btns, zip_btn],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/tmp"])

