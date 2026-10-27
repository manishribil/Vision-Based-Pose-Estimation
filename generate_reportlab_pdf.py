#!/usr/bin/env python3
"""
Generate Clean, Publication-Quality 2-Page Mini-Project PDF Write-up
Using ReportLab for Exact Academic Typography, Flawless Wrapping, and Crisp Vectors.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "PROJECT_WRITEUP.pdf"

def build_pdf():
    # Setup document with exact 0.4 in (28.8 pt) margins
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=26,
        bottomMargin=26
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#FFFFFF'),
        alignment=1 # Center
    )
    header_sub_style = ParagraphStyle(
        'HeaderSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#93C5FD'),
        alignment=1 # Center
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1E293B')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=4,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=3
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=9.2,
        textColor=colors.HexColor('#334155'),
        leftIndent=8,
        spaceAfter=2
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor('#1E293B')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    table_cell_green = ParagraphStyle(
        'TableCellGreen',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#047857')
    )
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.0,
        leading=8.5,
        textColor=colors.HexColor('#94A3B8'),
        alignment=1
    )

    story = []

    # =========================================================================
    # PAGE 1
    # =========================================================================

    # Title Banner Table
    title_data = [
        [Paragraph("UE24CS352A: MACHINE LEARNING MINI-PROJECT REPORT", header_sub_style)],
        [Paragraph("Vision-Based Precision Pose Estimation for Autonomous Formation Flying", header_title_style)]
    ]
    title_table = Table(title_data, colWidths=[556])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E293B')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 3))

    # Meta bar
    meta_data = [[
        Paragraph("<b>Team:</b> Manish Bala & Manasvi D R  |  <b>PES University</b>", meta_style),
        Paragraph("<b>Repo:</b> github.com/manishribil/Vision-Based-Pose-Estimation", ParagraphStyle('R', parent=meta_style, alignment=2, textColor=colors.HexColor('#0284C7')))
    ]]
    meta_table = Table(meta_data, colWidths=[316, 240])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('LINEBELOW', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))

    # Page 1: Two Column Layout Table
    # Col 1: Text sections (Width 272 pt) | Col 2: Image & Pipeline (Width 276 pt)
    col1_content = []
    col1_content.append(Paragraph("1. PROBLEM STATEMENT & MOTIVATION", section_heading))
    col1_content.append(Paragraph(
        "Autonomous multi-UAV formation flight is vital for cooperative search and rescue, perimeter monitoring, and collaborative cargo transport. Conventional swarms rely on Global Navigation Satellite Systems (GNSS) and inter-drone radio telemetry, which are acutely vulnerable to RF jamming, spoofing, multipath degradation, and complete denial in indoor or contested environments. This project develops a <b>passive onboard monocular vision pipeline</b> that detects peer drones, estimates 2D centroids and 3D relative pose, and continuously monitors formation health without transmitting radio signals.",
        body_style
    ))

    col1_content.append(Paragraph("2. DATASET SPECIFICATIONS", section_heading))
    col1_content.append(Paragraph(
        "Trained and benchmarked on the Roboflow <i>Multiple Drones</i> aerial formation dataset:",
        body_style
    ))
    col1_content.append(Paragraph("• <b>Target Drone:</b> 3DR Solo quadcopters (physical diagonal width <i>W</i> = 0.46 m).", bullet_style))
    col1_content.append(Paragraph("• <b>Resolution:</b> Native 640 × 640 RGB imagery captured from chase-drone perspectives.", bullet_style))
    col1_content.append(Paragraph("• <b>Test Split Scope:</b> 96 sequential flight frames with 277 annotated ground-truth drone instances.", bullet_style))
    col1_content.append(Paragraph("• <b>Dynamics:</b> Scale variation (box widths 8.4 px to 154.6 px), dynamic cloud/sky contrast, and multi-drone occlusion.", bullet_style))

    col1_content.append(Paragraph("3. METHODOLOGY & 10-PHASE ARCHITECTURE", section_heading))
    col1_content.append(Paragraph(
        "<b>Phase 1 & 2 (Detection & Centroid Extraction):</b> YOLO11n outputs bounding boxes [x1, y1, x2, y2]. Drone centroids are computed as <i>c_x = (x1+x2)/2</i>, <i>c_y = (y1+y2)/2</i>.",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 3 (Spatial Relations):</b> Calculates horizontal and vertical relative bearing between all pairs: <i>dx = x_B - x_A</i>, <i>dy = y_B - y_A</i>.",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 4 (Evaluation):</b> Benchmark validation on test split (Precision, Recall, mAP).",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 5 & 6 (Normalization & Mapping):</b> Centroids are normalized as <i>x_norm = c_x / W_img</i>, <i>y_norm = c_y / H_img</i> to ensure scale and resolution invariance across cameras.",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 7 & 8 (Pairwise Euclidean Spacing):</b> Evaluates pairwise normalized separation <i>dist = sqrt(dx_norm^2 + dy_norm^2)</i> across all pairs and summarizes swarm statistics.",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 9 (Distribution Visualization):</b> Visualizes formation density charts.",
        body_style
    ))
    col1_content.append(Paragraph(
        "<b>Phase 10 (Monocular 3D Pose & Formation Health Tracking):</b><br/>"
        "• <i>Pinhole Depth Estimation:</i> <b>Z = (f · W) / w</b>, where <i>w</i> is detected box width in pixels, <i>W = 0.46 m</i>, and <i>f = 554 px</i> (assumed 60° HFOV).<br/>"
        "• <i>Lateral Displacements:</i> <b>X = (c_x - c0) · Z / f</b>, <b>Y = (c_y - c0) · Z / f</b>.<br/>"
        "• <i>Dimensionless Spacing:</i> <b>s = d_px / w_bar</b> (in drone-widths). Classified into: "
        "<b>Maintained</b> (3.0 ≤ s ≤ 10.0), <b>Too Close</b> (s < 3.0), or <b>Too Far</b> (s > 10.0).<br/>"
        "• <i>Live Telemetry HUD & Radar:</i> Real-time status badge and top-down X-Z radar view.",
        body_style
    ))

    # Col 2: Visual Sample + Pipeline Box
    col2_content = []
    col2_content.append(Paragraph("FORMATION TRACKING & TELEMETRY HUD", section_heading))
    sample_frame_path = PROJECT_DIR / "results/distance/frames/00128.jpg"
    if sample_frame_path.exists():
        col2_content.append(Image(str(sample_frame_path), width=270, height=195))
        col2_content.append(Paragraph("<i>Figure 1: Phase 10 Output showing 4 tracked drones with 3D depths, inter-vehicle links, telemetry HUD banner, and top-down radar view inset.</i>", ParagraphStyle('Cap', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=6.5, leading=8.5, textColor=colors.HexColor('#64748B'))))
    
    col2_content.append(Spacer(1, 4))
    col2_content.append(Paragraph("SYSTEM PIPELINE STAGES", section_heading))

    pipeline_stages = [
        ("Step 1: Raw Video Stream", "640x640 Test Video Frames / Real-Time Chase Camera"),
        ("Step 2: YOLO11n Detector", "High-Recall Localization (2.6M parameters, >60 FPS)"),
        ("Step 3: Centroid Extraction", "Centroids (c_x, c_y) & Bearing Relationships"),
        ("Step 4: Scale Normalization", "Resolution-Invariant Formation Geometry Mapping"),
        ("Step 5: 3D Monocular Range", "Pinhole Depth Z = (f·W)/w | Lateral Offsets X, Y in meters"),
        ("Step 6: Formation Health Engine", "Dimensionless Nearest-Neighbor Spacing (Drone-Widths)"),
        ("Step 7: Telemetry HUD & Radar", "Live Status Badge & Bird's-Eye Top-Down Projection")
    ]
    p_data = []
    for s_title, s_desc in pipeline_stages:
        p_data.append([
            Paragraph(f"<b>{s_title}:</b> {s_desc}", ParagraphStyle('PStep', parent=styles['Normal'], fontName='Helvetica', fontSize=6.7, leading=8.6, textColor=colors.HexColor('#1E293B')))
        ])
    p_table = Table(p_data, colWidths=[270])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    col2_content.append(p_table)

    page1_grid = Table([[col1_content, col2_content]], colWidths=[278, 278])
    page1_grid.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(page1_grid)

    story.append(Spacer(1, 4))
    story.append(Paragraph("Page 1 of 2  |  UE24CS352A Machine Learning Mini-Project  |  PES University", footer_style))

    # =========================================================================
    # PAGE 2
    # =========================================================================
    story.append(PageBreak())

    # Mini header on Page 2
    p2_head = Table([[
        Paragraph("<b>UE24CS352A: Vision-Based Precision Pose Estimation</b>", ParagraphStyle('P2H', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=10, textColor=colors.HexColor('#1E293B'))),
        Paragraph("Experimental Results, Sensitivity Analysis & Conclusions", ParagraphStyle('P2R', parent=styles['Normal'], fontName='Helvetica', fontSize=8.0, leading=10, textColor=colors.HexColor('#64748B'), alignment=2))
    ]], colWidths=[316, 240])
    p2_head.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(p2_head)
    story.append(Spacer(1, 4))

    # Page 2: Top Grid: Quantitative Results Table (Left) + Error Sensitivity (Right)
    # Col 1: Results Table & Error Text | Col 2: Error Curve & Edge Cases
    col2_1 = []
    col2_1.append(Paragraph("4. DETECTION EVALUATION RESULTS", section_heading))
    res_data = [
        [Paragraph("<b>Metric</b>", table_cell_bold), Paragraph("<b>Score</b>", table_cell_bold), Paragraph("<b>Significance</b>", table_cell_bold)],
        [Paragraph("Precision", table_cell), Paragraph("99.63%", table_cell_green), Paragraph("Near-zero false alarms in operating domain", table_cell)],
        [Paragraph("Recall", table_cell), Paragraph("99.51%", table_cell_green), Paragraph("99.5% of drones detected across all test frames", table_cell)],
        [Paragraph("mAP @ 0.50", table_cell), Paragraph("99.36%", table_cell_green), Paragraph("Outstanding localization at standard threshold", table_cell)],
        [Paragraph("mAP @ 0.50:0.95", table_cell), Paragraph("85.12%", table_cell_green), Paragraph("High strictness across small & large scales", table_cell)],
        [Paragraph("Test Dataset Scope", table_cell), Paragraph("96 frames", table_cell_bold), Paragraph("277 ground-truth annotated drone targets", table_cell)]
    ]
    res_table = Table(res_data, colWidths=[68, 48, 154])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    col2_1.append(res_table)
    col2_1.append(Spacer(1, 4))

    col2_1.append(Paragraph("5. DEPTH ERROR & MATHEMATICAL SENSITIVITY", section_heading))
    col2_1.append(Paragraph(
        "<b>Ground-Truth Comparison (272 matched detections, mean IoU = 0.915):</b><br/>"
        "• <b>MAE:</b> 0.252 m  |  <b>RMSE:</b> 0.360 m  |  <b>Mean Error (MAPE):</b> 2.5%  |  <b>90th Pct:</b> 5.2%.<br/>"
        "<b>Mathematical Sensitivity Derivation:</b><br/>"
        "Differentiating <i>Z = (f · W) / w</i> with respect to pixel box width <i>w</i>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>dZ / dw = - (f · W) / w^2 = - Z / w</i><br/>"
        "In relative fractional terms: <b>| δZ / Z | = | δw / w |</b><br/>"
        "<b>Physical Implication:</b> Percentage depth error equals percentage box width error. For near drones (<i>w = 120 px</i>), a 1 px jitter causes only <b>0.83%</b> depth error. For distant drones (<i>w = 10 px</i>), 1 px jitter causes <b>10.0%</b> error. Thus, bounding-box regression precision dictates far-field range uncertainty.",
        body_style
    ))

    col2_2 = []
    err_plot_path = PROJECT_DIR / "results/distance/error_analysis.png"
    if err_plot_path.exists():
        col2_2.append(Paragraph("SENSITIVITY & ERROR DISTRIBUTION", section_heading))
        col2_2.append(Image(str(err_plot_path), width=270, height=88))
        col2_2.append(Paragraph("<i>Figure 2: Ground-truth distance error distribution and sensitivity curve (|error| ~ 1/w).</i>", ParagraphStyle('Cap2', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=6.2, leading=7.5, textColor=colors.HexColor('#64748B'))))

    col2_2.append(Spacer(1, 2))
    col2_2.append(Paragraph("6. EDGE-CASE STRESS TESTING & FAILURE MODES", section_heading))
    edge_data = [
        [Paragraph("<b>Condition</b>", table_cell_bold), Paragraph("<b>Conf</b>", table_cell_bold), Paragraph("<b>Empirical Behavior / Impact</b>", table_cell_bold)],
        [Paragraph("Motion Blur (K=15)", table_cell), Paragraph("0.11", table_cell), Paragraph("Significant confidence drop due to rotor smear", table_cell)],
        [Paragraph("Motion Blur (K=35)", table_cell), Paragraph("0.00", ParagraphStyle('F', parent=table_cell, textColor=colors.HexColor('#EF4444'))), Paragraph("Complete detection dropout (0 boxes found)", table_cell)],
        [Paragraph("Darkness (50% & 20%)", table_cell), Paragraph("0.35", table_cell_green), Paragraph("High silhouette contrast against sky preserves hits", table_cell)],
        [Paragraph("Boundary Crop (35%)", table_cell), Paragraph("0.31", table_cell), Paragraph("Truncated width artificially inflates estimated depth", table_cell)],
        [Paragraph("Urban Scene (bus.jpg)", table_cell), Paragraph("0.47", ParagraphStyle('W', parent=table_cell, textColor=colors.HexColor('#F59E0B'))), Paragraph("False positive on window textures (dataset sky bias)", table_cell)],
        [Paragraph("White Drone (drone.jpg)", table_cell), Paragraph("0.39", table_cell), Paragraph("Split partial boxes due to dark 3DR Solo training bias", table_cell)]
    ]
    edge_table = Table(edge_data, colWidths=[76, 28, 166])
    edge_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    col2_2.append(edge_table)

    top_grid_p2 = Table([[col2_1, col2_2]], colWidths=[278, 278])
    top_grid_p2.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(top_grid_p2)
    story.append(Spacer(1, 4))

    # Bottom Sections: Formation Status Breakdown + Conclusions + Contributions
    bot_content = []
    
    # 7. Formation Status Breakdown & Summary
    bot_content.append(Paragraph("7. FORMATION STATUS SUMMARY (96 TRAJECTORY FRAMES)", section_heading))
    f_stat_data = [
        [
            Paragraph("• <b>Maintained Formation:</b> 44 frames (45.8%)<br/>• <b>Too Far (Outside Limit):</b> 34 frames (35.4%)", body_style),
            Paragraph("• <b>Single Drone Tracked:</b> 12 frames (12.5%)<br/>• <b>Too Close (Collision Hazard):</b> 6 frames (6.3%)", body_style),
            Paragraph("• <b>Nearest-Neighbor Spacing:</b> Median 5.9 drone-widths (IQR 4.1 - 9.4)<br/>• <b>Depth Extent:</b> 1.6 m to 30.2 m (Median 8.5 m)", body_style)
        ]
    ]
    f_stat_table = Table(f_stat_data, colWidths=[180, 180, 196])
    f_stat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    bot_content.append(f_stat_table)
    bot_content.append(Spacer(1, 4))

    # 8. Conclusions & Individual Contributions
    bot_content.append(Paragraph("8. CONCLUSIONS & EQUITABLE INDIVIDUAL CONTRIBUTIONS", section_heading))
    contrib_text = (
        "<b>Conclusions & Practical Takeaways:</b> The project proves that high-precision relative pose estimation for autonomous formation flight is achievable via lightweight onboard monocular vision. Spacing in dimensionless drone-widths provides an immediate, calibration-free collision avoidance signal. Physical metre distance estimation is viable with known vehicle airframe width (MAE = 0.252 m). Future work includes EKF temporal filtering to eliminate far-field pixel jitter and negative-sample fine-tuning to eliminate urban false positives.<br/><br/>"
        "<b>Individual Contributions:</b><br/>"
        "• <b>Manish Bala:</b> YOLO11n object detector training & validation (Phases 1, 4); 2D centroid extraction pipeline (Phase 2); spatial bearing and orientation analysis (Phase 3); coordinate normalization & initial distance analysis (Phases 6, 7); core git repository setup.<br/>"
        "• <b>Manasvi D R:</b> Pinhole monocular 3D distance and lateral displacement formulation (Phase 10); nearest-neighbor dimensionless spacing formulation; dynamic formation health classifier (Maintained / Too Close / Outside Limit) & top-down radar projection; ground-truth error sensitivity & box jitter mathematical derivation (|δZ/Z| = |δw/w|); edge-case robustness test suite (<i>tools/edge_case_tests.py</i>); real camera focal calibration & error analysis tool (<i>tools/manasvi_distance_viz.py</i>); end-to-end master verification harness (<i>verify_pipeline.py</i>) and executive PDF report generator."
    )
    bot_content.append(Paragraph(contrib_text, body_style))

    story.extend(bot_content)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Page 2 of 2  |  UE24CS352A Machine Learning Mini-Project  |  PES University", footer_style))

    # Build the PDF
    doc.build(story)
    print(f"SUCCESS: ReportLab generated {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    build_pdf()
