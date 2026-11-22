"""PDF incident report using reportlab."""
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEV_COLORS = {"Critical": "#b91c1c", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#16a34a"}


def generate_pdf(facts: dict, sev: dict, ai: dict, source_name: str = "pasted logs") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    st = getSampleStyleSheet()
    P = lambda t, s="BodyText": Paragraph(escape(str(t)), st[s])
    bullets = lambda items: [P(f"• {i}") for i in items]
    el = [P("IncidentIQ - Incident Report", "Title"),
          P(f"Generated {datetime.now():%Y-%m-%d %H:%M} | Source: {source_name} | Analysis: {ai.get('source', '')}"), Spacer(1, 12)]
    t = Table([["Severity", "Errors", "Warnings", "Repeated patterns", "Status"],
               [sev["severity"].upper(), facts["errors"], facts["warnings"], facts["repeated_patterns"], sev["status"]]],
              hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor(SEV_COLORS[sev["severity"]])),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold")]))
    el += [t, Spacer(1, 12), P("Summary", "Heading2"), P(ai.get("summary", "")),
           P("Severity reasoning (rule-based)", "Heading2"), *bullets(sev["reasons"])]
    rv = ai.get("severity_review") or {}
    if rv:
        el.append(P(f"AI severity review: {'agrees' if rv.get('agrees') else 'suggests ' + str(rv.get('suggested'))}. {rv.get('justification', '')}"))
    el += [P("Evidence (present in logs)", "Heading2"), *bullets(ai.get("evidence", [])),
           P("Hypotheses (not confirmed)", "Heading2"),
           *bullets([f"{h['cause']} [confidence: {h['confidence']}] - {h['reasoning']}" for h in ai.get("hypotheses", [])]),
           P("Impact", "Heading2"), P(ai.get("impact", "")),
           P("Recommended actions", "Heading2"), *[P(f"{i}. {r}") for i, r in enumerate(ai.get("recommendations", []), 1)],
           P("Top error patterns", "Heading2")]
    rows = [["Count", "Level", "Service", "Message"]] + [[p["count"], p["level"], p["service"], Paragraph(escape(p["message"]), st["BodyText"])] for p in facts["patterns"][:10]]
    pt = Table(rows, colWidths=[1.5 * cm, 2 * cm, 3 * cm, 10 * cm], hAlign="LEFT")
    pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el.append(pt)
    doc.build(el)
    return buf.getvalue()
