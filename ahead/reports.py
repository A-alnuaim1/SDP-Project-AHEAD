"""Export an individual educational screening summary as a PDF."""

import math
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from ahead.config import DISEASES, pretty_label


def patient_report(disease, patient_id, values, score, threshold, model, metrics, status="", note="", metric_source="test",
                   metric_scope="", usual_threshold=None, metric_caveat=""):
    stream = BytesIO()
    doc = SimpleDocTemplate(stream, pagesize=A4, leftMargin=44, rightMargin=44)
    styles = getSampleStyleSheet()
    story = [Paragraph("AHEAD | Screening Summary", styles["Title"]), Spacer(1, 12)]
    lines = [
        f"Record: {patient_id}", f"Condition: {DISEASES[disease]['label']}",
        f"Model: {model}", f"Model screening score: {score * 100:.1f}%",
        f"Classification threshold: {threshold * 100:.1f}%" + (
            f" (lowered from {usual_threshold * 100:.1f}% because some inputs were left blank)"
            if usual_threshold is not None and threshold < usual_threshold else ""),
        f"Result: {'Flagged for review' if score >= threshold else 'Not flagged'}",
        f"Doctor review: {status or 'Pending'}",
    ]
    story.extend(Paragraph(_safe(line), styles["Normal"]) for line in lines)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Entered information", styles["Heading2"]))
    blank = [k for k, v in values.items() if _blank(v)]
    rows = [["Input", "Value"]] + [[pretty_label(k), "Not provided" if _blank(v) else str(v)] for k, v in values.items()]
    table = Table([[_safe(str(cell)) for cell in row] for row in rows], colWidths=[225, 275], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9F3F6")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    if blank:
        story.extend([Spacer(1, 6), Paragraph(
            "Inputs marked Not provided were left blank. The model filled them with typical values from its "
            "training data, so this estimate is less reliable than one with every value entered.", styles["Normal"])])
    story.extend([Spacer(1, 12), Paragraph("How to read this result", styles["Heading2"])])
    story.append(Paragraph(
        "This score is a model output, not a patient's true probability of disease. A flag is not a diagnosis; "
        "a lower score does not rule disease out. A qualified clinician should review symptoms, laboratory "
        "measurements and history independently.", styles["Normal"]))
    story.append(Paragraph(
        f"Model {metric_source}-set recall{metric_scope}: {metrics.get('recall_disease', 0) * 100:.1f}%. "
        f"Model {metric_source}-set precision{metric_scope}: {metrics.get('precision_disease', 0) * 100:.1f}%. "
        "These are dataset results and do not establish performance in a particular hospital.", styles["Normal"]))
    if metric_caveat:
        story.append(Paragraph(_safe(metric_caveat), styles["Normal"]))
    if note:
        story.extend([Spacer(1, 9), Paragraph("Doctor note", styles["Heading2"]), Paragraph(_safe(note), styles["Normal"])])
    story.extend([Spacer(1, 12), Paragraph("Questions to discuss with a healthcare professional", styles["Heading2"]),
                  Paragraph("Which findings should be confirmed? What follow-up tests or appointments are appropriate?", styles["Normal"])])
    doc.build(story)
    return stream.getvalue()


def _blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _safe(value):
    from xml.sax.saxutils import escape
    return escape(value)
