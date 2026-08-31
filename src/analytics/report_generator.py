"""End-of-session report generation (PNG and PDF).

Turns the logged session into a shareable one-page artifact: a gesture-frequency
bar chart, an interaction heatmap over the frame area, total objects created,
and session length.  Matplotlib renders the charts; an optional PDF writer
(fpdf2 or reportlab) packages them.  If the optional deps are missing this
module degrades gracefully to text-only summaries rather than crashing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.logger import get_logger

log = get_logger("analytics.report")


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def generate_report(
    events,
    duration: float,
    report_dir: str | Path = "output/reports",
    heatmap_size: int = 32,
    title: str = "GestureForge Session Report",
) -> dict[str, str]:
    """Generate PNG + optional PDF reports from logged *events*.

    Args:
        events: List of rows ``(event_type, detail, hand_index, x, y)`` as
            produced by :meth:`SessionLogger.retrieve_events`.
        duration: Session length in seconds.
        report_dir: Where to write the output files.
        heatmap_size: Resolution of the heatmap grid.
        title: Title for the report.

    Returns:
        A dict of generated file paths (e.g. ``{"png": ..., "pdf": ...}``).
    """
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    event_types = [str(r[0]) for r in events]
    counts = Counter(event_types)

    # Build gesture-frequency bar chart + heatmap into a single PNG figure.
    heatmap = _build_heatmap(events, heatmap_size)
    png_path = out_dir / "session_report.png"
    _render_png(title, duration, counts, heatmap, png_path, list(event_types))
    generated["png"] = str(png_path)

    # Optional PDF packaging.
    pdf_path = out_dir / "session_report.pdf"
    if not _attempt_pdf(png_path, duration, counts, pdf_path, title):
        # Fall back to a plain-text summary.
        txt_path = out_dir / "session_report.txt"
        txt_path.write_text(_text_summary(title, duration, counts), encoding="utf-8")
        generated["txt"] = str(txt_path)
    else:
        generated["pdf"] = str(pdf_path)

    return generated


def _build_heatmap(events, heatmap_size: int) -> np.ndarray:
    grid = np.zeros((heatmap_size, heatmap_size), dtype=np.float64)
    for r in events:
        x, y = r[3], r[4]
        if x is None or y is None:
            continue
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            ix = min(heatmap_size - 1, int(x * heatmap_size))
            iy = min(heatmap_size - 1, int(y * heatmap_size))
            grid[iy, ix] += 1.0
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def _render_png(
    title: str,
    duration: float,
    counts: Counter,
    heatmap: np.ndarray,
    path: Path,
    event_types: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.suptitle(f"{title}\nSession length: {duration/60:.1f} min", fontweight="bold")

    labels = list(counts.keys())
    values = list(counts.values())
    if labels:
        ax1.bar(range(len(labels)), values, color="#5b8def")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Gesture / Event Frequency")
        ax1.grid(True, axis="y", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No events logged", ha="center", va="center")

    im = ax2.imshow(heatmap, cmap="viridis", origin="lower", aspect="auto")
    ax2.set_title("Interaction Heatmap (frame area)")
    ax2.set_xlabel("frame x")
    ax2.set_ylabel("frame y")
    fig.colorbar(im, ax=ax2, shrink=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _ = event_types


def _attempt_pdf(
    png_path: Path, duration: float, counts: Counter, pdf_path: Path, title: str
) -> bool:
    if _has("fpdf"):
        try:
            from fpdf import FPDF  # type: ignore

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Session length: {duration/60:.1f} minutes", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(png_path), w=170)
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Event breakdown:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            for label, value in counts.most_common():
                pdf.cell(0, 7, f"  {label}: {value}", new_x="LMARGIN", new_y="NEXT")
            pdf.output(str(pdf_path))
            log.info("Wrote PDF report to %s", pdf_path)
            return True
        except Exception as exc:
            log.warning("PDF (fpdf) failed: %s", exc)
            return False
    if _has("reportlab"):
        try:
            from reportlab.lib.pagesizes import letter  # type: ignore
            from reportlab.pdfgen import canvas as _canvas  # type: ignore

            c = _canvas.Canvas(str(pdf_path), pagesize=letter)
            w, h = letter
            c.setFont("Helvetica-Bold", 18)
            c.drawString(60, h - 60, title)
            c.setFont("Helvetica", 11)
            c.drawString(60, h - 90, f"Session length: {duration/60:.1f} minutes")
            c.drawImage(str(png_path), 60, h - 500, width=480, height=220)
            c.setFont("Helvetica-Bold", 13)
            c.drawString(60, 180, "Event breakdown:")
            c.setFont("Helvetica", 11)
            yy = 160
            for label, value in counts.most_common():
                c.drawString(70, yy, f"  {label}: {value}")
                yy -= 16
            c.save()
            log.info("Wrote PDF report to %s", pdf_path)
            return True
        except Exception as exc:
            log.warning("PDF (reportlab) failed: %s", exc)
            return False
    log.warning("No PDF writer available; emitting text summary instead")
    return False


def _text_summary(title: str, duration: float, counts: Counter) -> str:
    lines = [f"{title}", f"Session length: {duration/60:.1f} minutes", ""]
    for label, value in counts.most_common():
        lines.append(f"  {label}: {value}")
    return "\n".join(lines) + "\n"
