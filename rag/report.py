from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import io
from datetime import datetime

def generate_pdf_report(
    patient_name: str,
    doc_name: str,
    alertes: list,
    conversation: list
) -> bytes:
    """
    Génère un rapport PDF téléchargeable.
    Retourne les bytes du PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    content = []

    # En-tête
    titre_style = ParagraphStyle(
        "titre", fontSize=18, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1565C0"), spaceAfter=6
    )
    content.append(Paragraph("🏥 MediRAG — Rapport d'Analyse Médicale", titre_style))
    content.append(Paragraph(
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        styles["Normal"]
    ))
    content.append(Paragraph(f"Document analysé : {doc_name}", styles["Normal"]))
    if patient_name:
        content.append(Paragraph(f"Patient : {patient_name}", styles["Normal"]))
    content.append(Spacer(1, 0.5*cm))

    # Alertes
    if alertes:
        content.append(Paragraph("⚠️ Valeurs Anormales Détectées",
                                  ParagraphStyle("h2", fontSize=14,
                                  fontName="Helvetica-Bold",
                                  textColor=colors.red, spaceAfter=4)))

        table_data = [["Paramètre", "Valeur", "Norme", "Statut"]]
        for alerte in alertes:
            table_data.append([
                alerte.get("parametre", ""),
                alerte.get("valeur", ""),
                alerte.get("norme", ""),
                "🔴 Critique" if alerte.get("niveau") == "critique" else "🟡 Élevé"
            ])

        table = Table(table_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F5F5F5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        content.append(table)
        content.append(Spacer(1, 0.5*cm))

    # Questions et réponses
    if conversation:
        content.append(Paragraph("💬 Vos Questions et Réponses",
                                  ParagraphStyle("h2", fontSize=14,
                                  fontName="Helvetica-Bold", spaceAfter=4)))
        for msg in conversation:
            if msg["role"] == "user":
                content.append(Paragraph(
                    f"<b>Question :</b> {msg['content']}",
                    styles["Normal"]
                ))
            else:
                content.append(Paragraph(
                    f"<b>Réponse MediRAG :</b> {msg['content']}",
                    styles["Normal"]
                ))
            content.append(Spacer(1, 0.2*cm))

    # Avertissement
    content.append(Spacer(1, 1*cm))
    content.append(Paragraph(
        "⚕️ Ce rapport est fourni à titre informatif uniquement. "
        "Consultez toujours un professionnel de santé pour un avis médical.",
        ParagraphStyle("warning", fontSize=9,
                       textColor=colors.HexColor("#E65100"))
    ))

    doc.build(content)
    return buffer.getvalue()