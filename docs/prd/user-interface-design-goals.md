# User Interface Design Goals

## Overall UX Vision
The interface prioritizes clarity and efficiency for technical users (AI safety researchers and evaluation teams). The primary CLI provides transparent progress tracking with real-time statistics, while the optional Streamlit demo offers visual exploration of bias detection results. All interactions emphasize reproducibility and auditability over aesthetic polish.

## Key Interaction Paradigms
Command-line driven batch processing with clear progress indicators, checkpoint recovery for interrupted runs, and comprehensive logging. The optional web interface provides read-only visualization of results with interactive filtering and statistical exploration. All outputs include confidence intervals and methodological transparency.

## Core Screens and Views
• **CLI Progress Display**: Real-time processing status with question count, model consensus, and running statistics
• **Results Summary Terminal Output**: Comprehensive bias report with exploitability percentages, statistical metrics, and confidence intervals
• **Streamlit Dashboard** (optional): Interactive visualization of removed vs retained questions, bias distribution charts, and model agreement heatmaps
• **Audit Log Viewer**: Detailed trace of all decisions for reproducibility verification

## Accessibility: None
Technical tool for expert users; standard terminal accessibility applies

## Branding
No specific branding requirements - focus on clear, professional presentation of technical data

## Target Device and Platforms: Desktop Only
Linux command-line environment (Ubuntu 20.04+) with optional web browser for Streamlit dashboard
