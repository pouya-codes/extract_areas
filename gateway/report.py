"""PDF report of a finished job: a summary, one table row per region and,
optionally, each region's overlay.

The report is rendered from the same job view the poll endpoint returns, so it
shows exactly what the client would see. Overlays expire before the job record
does (``OVERLAY_TTL_SECONDS`` < ``JOB_TTL_SECONDS``); a region whose overlay is
gone is listed with a note rather than failing the whole report.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Optional
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from inference_platform import __version__
from inference_platform.contracts import TaskStatus

logger = logging.getLogger(__name__)

OverlayLoader = Callable[[str, str], Optional[bytes]]

PAGE_SIZE = letter
MARGIN = 0.5 * inch
CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN
OVERLAY_MAX_HEIGHT = 6.0 * inch
# Overlays are downscaled before embedding; a 500-region job at full
# resolution would produce a PDF nobody can open.
OVERLAY_MAX_PIXELS = 1600
MAX_TEXT = 600
MAX_SCORE_ITEMS = 12

STATUS_COLORS = {
    TaskStatus.DONE: colors.HexColor("#1b7f3b"),
    TaskStatus.FAILED: colors.HexColor("#b3261e"),
    TaskStatus.CANCELLED: colors.HexColor("#6b6b6b"),
    TaskStatus.RUNNING: colors.HexColor("#1a5fb4"),
    TaskStatus.PENDING: colors.HexColor("#6b6b6b"),
}


def report_filename(view: dict[str, Any]) -> str:
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", view.get("model_name") or "job").strip("_") or "job"
    return f"{model}-{str(view['job_id'])[:8]}-results.pdf"


def build_job_report(view: dict[str, Any], load_overlay: Optional[OverlayLoader] = None) -> bytes:
    """Render ``view`` (a :meth:`JobStore.job_view` with regions) as PDF bytes.

    ``load_overlay(job_id, task_key)`` supplies overlay PNGs; pass ``None`` to
    leave the images out.
    """
    styles = _styles()
    regions = view.get("regions") or []
    story: list[Any] = [
        Paragraph("Inference batch report", styles["Title"]),
        _summary_table(view, styles),
        Spacer(1, 0.25 * inch),
        Paragraph(f"Region results ({len(regions)})", styles["Heading2"]),
        _results_table(regions, styles),
    ]
    if load_overlay is not None:
        story.extend(_overlay_section(view, regions, load_overlay, styles))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=0.75 * inch,
        title=f"Inference results {view['job_id']}",
        author="AIM inference gateway",
        subject=f"{view.get('model_name', '')} on {view.get('slide_path', '')}",
        creator=f"AIM inference gateway {__version__}",
    )
    footer = f"Job {view['job_id']}"
    doc.build(story, onFirstPage=_footer(footer), onLaterPages=_footer(footer))
    return buffer.getvalue()


# ---------------------------------------------------------------------- #
# Sections
# ---------------------------------------------------------------------- #
def _summary_table(view: dict[str, Any], styles) -> Table:
    owner = view.get("username") or (
        f"user {view['user_id']}" if view.get("user_id") is not None else "-"
    )
    counts = (
        f"{view.get('regions_done', 0)} done, {view.get('regions_failed', 0)} failed, "
        f"{view.get('regions_cancelled', 0)} cancelled of {view.get('regions_total', 0)}"
    )
    rows = [
        ("Job ID", view.get("job_id")),
        ("Model", view.get("model_name")),
        ("Slide", view.get("slide_path")),
        ("Image ID", view.get("image_id")),
        ("Submitted by", owner),
        ("Status", f"{view.get('status')} - {view.get('message') or ''}"),
        ("Regions", counts),
        ("Submitted", _fmt_time(view.get("created_at"))),
        ("Completed", _fmt_time(view.get("completed_at"))),
        ("Downsample", _fmt_value(view.get("downsample"))),
        ("Hyperparameters", _fmt_mapping(view.get("hyperparameters")) or "defaults"),
    ]
    if view.get("client_ref"):
        rows.append(("Client reference", view["client_ref"]))
    if view.get("error"):
        rows.append(("First error", view["error"]))

    data = [
        [Paragraph(escape(label), styles["Label"]), _para(value, styles["Cell"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=[1.4 * inch, CONTENT_WIDTH - 1.4 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _results_table(regions: list[dict[str, Any]], styles) -> LongTable:
    header = ["#", "Region", "Label", "Status", "Result", "Scores", "Time (s)"]
    widths = [0.3, 1.1, 0.8, 0.7, 2.0, 2.0, 0.6]
    data: list[list[Any]] = [[Paragraph(h, styles["Header"]) for h in header]]
    style_cmds: list[tuple] = []
    for index, region in enumerate(regions, start=1):
        status = region.get("status") or ""
        result = region.get("str_result") or ""
        if region.get("error"):
            result = f"{result}\nError: {region['error']}" if result else f"Error: {region['error']}"
        seconds = (region.get("timing") or {}).get("model_seconds")
        data.append([
            _para(index, styles["Cell"]),
            _para(region.get("key"), styles["Cell"]),
            _para(region.get("label_slug") or "-", styles["Cell"]),
            _para(status, styles["Cell"]),
            _para(result or "-", styles["Cell"]),
            _para(_fmt_scores(region.get("score")) or "-", styles["Cell"]),
            _para(_fmt_value(seconds) if seconds is not None else "-", styles["Cell"]),
        ])
        if status in STATUS_COLORS:
            style_cmds.append(("TEXTCOLOR", (3, index), (3, index), STATUS_COLORS[status]))
        if index % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#f5f6f8")))
    if not regions:
        data.append([_para("No regions in this job.", styles["Cell"])] + [""] * (len(header) - 1))
        style_cmds.append(("SPAN", (0, 1), (-1, 1)))

    table = LongTable(data, colWidths=[w * inch for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3b52")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8ccd4")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        *style_cmds,
    ]))
    return table


def _overlay_section(
    view: dict[str, Any],
    regions: list[dict[str, Any]],
    load_overlay: OverlayLoader,
    styles,
) -> list[Any]:
    with_overlay = [
        r for r in regions if r.get("status") == TaskStatus.DONE and r.get("has_overlay")
    ]
    if not with_overlay:
        return []
    flowables: list[Any] = [PageBreak(), Paragraph("Region overlays", styles["Heading2"])]
    for region in with_overlay:
        title = Paragraph(
            escape(f"{region['key']}" + (f"  ({region['label_slug']})" if region.get("label_slug") else "")),
            styles["Heading4"],
        )
        caption = _para(region.get("str_result") or "", styles["Cell"])
        image = _overlay_image(view["job_id"], region["key"], load_overlay)
        body = image if image is not None else Paragraph(
            "Overlay no longer available (overlays expire before the job record).",
            styles["Note"],
        )
        flowables.append(KeepTogether([title, caption, Spacer(1, 4), body, Spacer(1, 0.2 * inch)]))
    return flowables


def _overlay_image(job_id: str, task_key: str, load_overlay: OverlayLoader) -> Optional[Image]:
    try:
        data = load_overlay(job_id, task_key)
        if not data:
            return None
        with PILImage.open(BytesIO(data)) as source:
            source.load()
            picture = source.convert("RGBA") if source.mode not in ("RGB", "RGBA") else source.copy()
        picture.thumbnail((OVERLAY_MAX_PIXELS, OVERLAY_MAX_PIXELS), PILImage.Resampling.LANCZOS)
        encoded = BytesIO()
        picture.save(encoded, format="PNG", optimize=True)
        encoded.seek(0)
    except Exception:  # noqa: BLE001 - one unreadable overlay must not sink the report
        logger.warning("overlay %s/%s could not be embedded", job_id, task_key, exc_info=True)
        return None

    width_px, height_px = picture.size
    scale = min(CONTENT_WIDTH / width_px, OVERLAY_MAX_HEIGHT / height_px)
    return Image(encoded, width=width_px * scale, height=height_px * scale, mask="auto")


# ---------------------------------------------------------------------- #
# Formatting
# ---------------------------------------------------------------------- #
def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cell", parent=base["BodyText"], fontSize=8, leading=10))
    base.add(ParagraphStyle("Label", parent=base["Cell"], fontName="Helvetica-Bold"))
    base.add(ParagraphStyle("Header", parent=base["Label"], textColor=colors.white))
    base.add(ParagraphStyle("Note", parent=base["Cell"], textColor=colors.HexColor("#6b6b6b")))
    return base


def _footer(text: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b6b6b"))
        canvas.drawString(MARGIN, 0.4 * inch, text)
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _para(value: Any, style) -> Paragraph:
    text = "" if value is None else str(value)
    if len(text) > MAX_TEXT:
        text = text[: MAX_TEXT - 3] + "..."
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + _fmt_mapping(value, sep=", ") + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    return str(value)


def _fmt_mapping(mapping: Any, sep: str = "\n") -> str:
    if not isinstance(mapping, dict) or not mapping:
        return ""
    return sep.join(f"{k}: {_fmt_value(v)}" for k, v in mapping.items())


def _fmt_scores(scores: Any) -> str:
    if isinstance(scores, dict):
        items = list(scores.items())
        lines = [f"{k}: {_fmt_value(v)}" for k, v in items[:MAX_SCORE_ITEMS]]
        if len(items) > MAX_SCORE_ITEMS:
            lines.append(f"... {len(items) - MAX_SCORE_ITEMS} more")
        return "\n".join(lines)
    if scores is None or scores == [] or scores == "":
        return ""
    return _fmt_value(scores)


def _fmt_time(value: Optional[str]) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except ValueError:
        return value
