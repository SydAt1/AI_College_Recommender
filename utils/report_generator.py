import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import json
from datetime import datetime
from typing import Dict, List, Any

def generate_pdf_report(recommendation_result: Dict[str, Any], user_profile: Dict[str, Any]) -> bytes:
    """Generate a PDF report of college recommendations"""
    
    # Create a buffer to store the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = styles['Normal']
    
    # Title
    story.append(Paragraph("AI College Recommender Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 20))
    
    # User Profile Section
    story.append(Paragraph("Student Profile", heading_style))
    
    profile_data = [
        ["Academic Information", ""],
        ["GPA", f"{user_profile.get('gpa', 'N/A')}"],
        ["SAT Score", f"{user_profile.get('sat_score', 'N/A')}"],
        ["ACT Score", f"{user_profile.get('act_score', 'N/A')}"],
        ["Class Rank Percentile", f"{user_profile.get('class_rank_percentile', 'N/A')}%"],
        ["", ""],
        ["Preferences", ""],
        ["Preferred States", ", ".join(user_profile.get('preferred_states', [])) or "Any"],
        ["Campus Size", user_profile.get('campus_size_preference', 'Any')],
        ["Location Type", user_profile.get('location_type_preference', 'Any')],
        ["Institution Type", user_profile.get('public_private_preference', 'Any')],
        ["", ""],
        ["Financial Information", ""],
        ["Maximum Budget", f"${user_profile.get('budget_max', 0):,}"],
        ["Need Financial Aid", "Yes" if user_profile.get('need_financial_aid') else "No"],
        ["Intended Major", user_profile.get('intended_major', 'Not specified')]
    ]
    
    profile_table = Table(profile_data, colWidths=[2*inch, 3*inch])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(profile_table)
    story.append(Spacer(1, 20))
    
    # Model Information
    model_info = recommendation_result.get('model_info', {})
    story.append(Paragraph("Model Information", heading_style))
    model_data = [
        ["Model Type", model_info.get('model_type', 'N/A')],
        ["Total Colleges Analyzed", f"{model_info.get('total_colleges', 0):,}"],
        ["Features Used", model_info.get('features_used', 0)],
        ["Processing Time", f"{recommendation_result.get('processing_time', 0)} seconds"]
    ]
    
    model_table = Table(model_data, colWidths=[2*inch, 3*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Recommendations Section
    story.append(Paragraph(f"College Recommendations ({len(recommendation_result.get('recommendations', []))} colleges)", heading_style))
    
    recommendations = recommendation_result.get('recommendations', [])
    
    for i, rec in enumerate(recommendations):
        # College header
        college_title = f"{rec['rank']}. {rec['name']}"
        story.append(Paragraph(college_title, styles['Heading3']))
        
        # College details table
        college_data = [
            ["Location", f"{rec['state']} • {rec['campus_size']} • {rec['location_type']}"],
            ["Institution Type", rec['public_private']],
            ["Match Score", f"{rec['confidence_score']:.1%}"],
            ["Acceptance Rate", f"{rec['acceptance_rate']:.1%}"],
            ["Graduation Rate", f"{rec['graduation_rate']:.1%}"],
            ["Average SAT", rec['avg_sat']],
            ["Average ACT", rec['avg_act']],
            ["Average GPA", rec['avg_gpa']],
            ["Student Population", f"{rec['student_population']:,}"],
            ["Cost (In-State)", f"${rec['total_cost_in_state']:,}"],
            ["Cost (Out-State)", f"${rec['total_cost_out_state']:,}"],
            ["Quality Score", f"{rec['quality_score']:.3f}"],
            ["Affordability Score", f"{rec['affordability_score']:.3f}"]
        ]
        
        college_table = Table(college_data, colWidths=[1.5*inch, 3.5*inch])
        college_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        
        story.append(college_table)
        story.append(Spacer(1, 15))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("Generated by AI College Recommender", styles['Normal']))
    story.append(Paragraph("This report is for informational purposes only. Please verify all information with official college sources.", styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    
    # Get the PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

def generate_csv_report(recommendation_result: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
    """Generate a CSV report of college recommendations"""
    
    recommendations = recommendation_result.get('recommendations', [])
    
    if not recommendations:
        return ""
    
    # Create DataFrame for recommendations
    df_recommendations = pd.DataFrame(recommendations)
    
    # Add user profile information
    profile_data = {
        'User GPA': [user_profile.get('gpa', 'N/A')],
        'User SAT Score': [user_profile.get('sat_score', 'N/A')],
        'User ACT Score': [user_profile.get('act_score', 'N/A')],
        'User Class Rank Percentile': [user_profile.get('class_rank_percentile', 'N/A')],
        'Preferred States': [', '.join(user_profile.get('preferred_states', [])) or 'Any'],
        'Campus Size Preference': [user_profile.get('campus_size_preference', 'Any')],
        'Location Type Preference': [user_profile.get('location_type_preference', 'Any')],
        'Institution Type Preference': [user_profile.get('public_private_preference', 'Any')],
        'Maximum Budget': [f"${user_profile.get('budget_max', 0):,}"],
        'Need Financial Aid': ['Yes' if user_profile.get('need_financial_aid') else 'No'],
        'Intended Major': [user_profile.get('intended_major', 'Not specified')]
    }
    
    df_profile = pd.DataFrame(profile_data)
    
    # Add model information
    model_info = recommendation_result.get('model_info', {})
    model_data = {
        'Model Type': [model_info.get('model_type', 'N/A')],
        'Total Colleges Analyzed': [model_info.get('total_colleges', 0)],
        'Features Used': [model_info.get('features_used', 0)],
        'Processing Time (seconds)': [recommendation_result.get('processing_time', 0)],
        'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    df_model = pd.DataFrame(model_data)
    
    # Create a buffer for the CSV
    buffer = BytesIO()
    
    # Write user profile section
    buffer.write(b"=== USER PROFILE ===\n")
    df_profile.to_csv(buffer, index=False)
    buffer.write(b"\n\n")
    
    # Write model information section
    buffer.write(b"=== MODEL INFORMATION ===\n")
    df_model.to_csv(buffer, index=False)
    buffer.write(b"\n\n")
    
    # Write recommendations section
    buffer.write(b"=== COLLEGE RECOMMENDATIONS ===\n")
    df_recommendations.to_csv(buffer, index=False)
    
    # Get the CSV content
    csv_content = buffer.getvalue().decode('utf-8')
    buffer.close()
    
    return csv_content

def generate_summary_report(recommendation_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary report with key statistics"""
    
    recommendations = recommendation_result.get('recommendations', [])
    
    if not recommendations:
        return {}
    
    # Calculate statistics
    confidence_scores = [rec['confidence_score'] for rec in recommendations]
    acceptance_rates = [rec['acceptance_rate'] for rec in recommendations]
    costs_in_state = [rec['total_cost_in_state'] for rec in recommendations]
    costs_out_state = [rec['total_cost_out_state'] for rec in recommendations]
    graduation_rates = [rec['graduation_rate'] for rec in recommendations]
    
    summary = {
        'total_recommendations': len(recommendations),
        'average_confidence_score': sum(confidence_scores) / len(confidence_scores),
        'average_acceptance_rate': sum(acceptance_rates) / len(acceptance_rates),
        'average_cost_in_state': sum(costs_in_state) / len(costs_in_state),
        'average_cost_out_state': sum(costs_out_state) / len(costs_out_state),
        'average_graduation_rate': sum(graduation_rates) / len(graduation_rates),
        'cost_range_in_state': {
            'min': min(costs_in_state),
            'max': max(costs_in_state)
        },
        'cost_range_out_state': {
            'min': min(costs_out_state),
            'max': max(costs_out_state)
        },
        'states_represented': list(set([rec['state'] for rec in recommendations])),
        'campus_sizes': list(set([rec['campus_size'] for rec in recommendations])),
        'institution_types': list(set([rec['public_private'] for rec in recommendations]))
    }
    
    return summary 