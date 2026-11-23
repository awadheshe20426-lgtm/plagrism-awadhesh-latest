import csv
import io
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from app.models.document import Document
from app.models.batch import Batch

class ReportService:
    """Service for generating reports (PDF, CSV)"""

    @classmethod
    def generate_csv_report(cls, documents: List[Document],
                            top_matches: Dict[str, Dict[str, Any]] = None) -> str:
        """
        Generate CSV report for a list of documents.
        top_matches: {document_id: {"similarity", "matched_filename", "matched_owner"}}
        computed from the stored comparisons table (the authoritative source).
        """
        top_matches = top_matches or {}
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(['Filename', 'Status', 'AI Score', 'Is AI?',
                         'Plagiarism Score', 'Top Match', 'Match Owner'])

        for doc in documents:
            match = top_matches.get(str(doc.id), {})
            writer.writerow([
                doc.filename,
                doc.status,
                f"{doc.ai_score:.2f}" if doc.ai_score is not None else "N/A",
                "Yes" if doc.is_ai_generated else "No",
                f"{match.get('similarity', 0.0):.2f}",
                match.get('matched_filename', '-'),
                match.get('matched_owner', '-'),
            ])

        return output.getvalue()

    @staticmethod
    def generate_pdf_report(batch: Batch, documents: List[Document],
                            top_matches: Dict[str, Dict[str, Any]] = None) -> bytes:
        """Generate PDF report for a batch, including plagiarism top-match per document."""
        top_matches = top_matches or {}
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['Title']
        story.append(Paragraph(f"Analysis Report - Batch {batch.id}", title_style))
        story.append(Spacer(1, 12))

        # Summary
        story.append(Paragraph(f"Total Documents: {len(documents)}", styles['Normal']))
        story.append(Spacer(1, 24))

        # Table Data
        data = [['Filename', 'AI Score', 'Verdict', 'Plagiarism', 'Top Match']]
        for doc_item in documents:
            score = f"{doc_item.ai_score:.1%}" if doc_item.ai_score is not None else "N/A"
            verdict = "AI-Generated" if doc_item.is_ai_generated else "Human-Written"
            if doc_item.ai_score is None:
                verdict = "Pending/Error"

            match = top_matches.get(str(doc_item.id), {})
            plag = f"{match.get('similarity', 0.0):.1%}"
            top = match.get('matched_filename', '-')
            owner = match.get('matched_owner')
            if owner:
                top = f"{top} ({owner})"

            data.append([doc_item.filename, score, verdict, plag, top])

        # Table Style
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        story.append(table)
        doc.build(story)
        
        return buffer.getvalue()
