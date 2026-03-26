#!/usr/bin/env python3
"""
Simple Markdown to PDF converter using reportlab
No external system dependencies required
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from pathlib import Path
import re

def convert_markdown_to_pdf():
    """Convert markdown to PDF using reportlab"""
    
    md_file = "B_TECH_PROJECT_REPORT.md"
    pdf_file = "B_TECH_PROJECT_REPORT.pdf"
    
    print("=" * 60)
    print("Markdown to PDF Converter")
    print("=" * 60)
    print()
    print(f"📄 Reading: {md_file}")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20,
        title="Real-Time Facial Emotion Detection - B.Tech Project Report"
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1a5fa0'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1a5fa0'),
        spaceAfter=12,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    # Process markdown
    lines = markdown_content.split('\n')
    section_count = 0
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        if not line.strip():
            continue
        
        # Main title
        if line.startswith('# ') and not line.startswith('## '):
            title = line[2:].strip()
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.3*inch))
        
        # Section headings
        elif line.startswith('## '):
            heading = line[3:].strip()
            section_count += 1
            if section_count > 1 and story:  # Page break before each section
                story.append(PageBreak())
            story.append(Paragraph(heading, heading1_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Subsection headings
        elif line.startswith('### '):
            heading = line[4:].strip()
            story.append(Paragraph(heading, heading2_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Bullet points
        elif line.lstrip().startswith('* ') or line.lstrip().startswith('- '):
            text = line.lstrip()[2:].strip()
            bullet_style = ParagraphStyle(
                'Bullet',
                parent=body_style,
                leftIndent=30,
                firstLineIndent=-15
            )
            story.append(Paragraph(f'• {text}', bullet_style))
        
        # Regular text
        else:
            cleaned_line = line.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
            if cleaned_line.strip():
                story.append(Paragraph(cleaned_line, body_style))
    
    # Build PDF
    print(f"🔨 Building PDF...")
    doc.build(story)
    
    file_size = Path(pdf_file).stat().st_size / 1024
    print()
    print(f"✅ Successfully converted!")
    print(f"📁 Output file: {pdf_file}")
    print(f"📊 File size: {file_size:.1f} KB")
    print()
    print(f"✨ Your B.Tech project report is ready for submission!")
    print()

if __name__ == "__main__":
    try:
        convert_markdown_to_pdf()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
