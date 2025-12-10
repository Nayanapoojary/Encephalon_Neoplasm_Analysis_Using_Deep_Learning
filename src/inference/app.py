"""
Complete Brain Tumor Diagnosis System
CLEAN MODERN DESIGN - Teal/White Theme with High Contrast
ResNet18 Architecture | 98.10% Accuracy
"""

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

# Add paths
current_dir = os.path.dirname(__file__)
# Prefer utils directory located alongside 'inference' (../utils)
utils_path = os.path.abspath(os.path.join(current_dir, '..', 'utils'))
# Fallbacks for other common layouts
alt_path = os.path.abspath(os.path.join(current_dir, 'src', 'utils'))
root_utils = os.path.abspath(os.path.join(current_dir, '..', '..', 'utils'))

if os.path.isdir(utils_path):
    sys.path.insert(0, utils_path)
elif os.path.isdir(alt_path):
    sys.path.insert(0, alt_path)
elif os.path.isdir(root_utils):
    sys.path.insert(0, root_utils)
else:
    # Leave sys.path unchanged; import will raise a helpful error caught below
    pass

try:
    from complete_diagnosis import BrainTumorDiagnosisSystem
    from severity_assessment import TumorSeverityAssessment
    from mri_validator import MRIValidator
except ImportError as e:
    st.error(f"❌ Import error: {e}\nMake sure the 'utils' package is located under src/utils or ../utils relative to this file.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Encephalon AI | Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CLEAN MODERN CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Clean background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf2 100%);
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Premium header */
    .hero-header {
        background: linear-gradient(135deg, #0d7377 0%, #14b8a6 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(13, 115, 119, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .hero-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.15rem;
        font-weight: 500;
        margin: 1rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Clean cards */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(13, 115, 119, 0.1);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .card:active::before {
        width: 300px;
        height: 300px;
    }
    
    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(13, 115, 119, 0.15);
        border-color: #14b8a6;
    }
    
    .card:active {
        transform: translateY(-4px) scale(0.98);
        transition: all 0.1s ease;
    }
    
    .card h3, .card h4 {
        color: #0d7377;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Severity display */
    .severity-box {
        background: #ffffff;
        border: 3px solid;
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .severity-box::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .severity-box:active::after {
        width: 400px;
        height: 400px;
    }
    
    .severity-box:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
    }
    
    .severity-box:active {
        transform: scale(0.98);
    }
    
    .severity-low { 
        border-color: #10b981; 
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .severity-moderate { 
        border-color: #f59e0b; 
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .severity-high { 
        border-color: #ef4444; 
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    .severity-critical { 
        border-color: #dc2626; 
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    }
    
    .severity-box h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
        color: inherit;
    }
    
    .severity-score {
        font-size: 5rem;
        font-weight: 900;
        margin: 1rem 0;
        color: inherit;
    }
    
    .severity-grade {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
        color: inherit;
    }
    
    .severity-confidence {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* Metrics */
    .metric-card {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.75rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(20, 184, 166, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        border-color: #14b8a6;
        box-shadow: 0 8px 25px rgba(13, 115, 119, 0.15);
        transform: translateY(-8px) rotate(2deg);
    }
    
    .metric-card:active {
        transform: translateY(-4px) scale(0.95);
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        color: #0d7377;
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .metric-subtitle {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Timeline */
    .timeline-item {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-left: 5px solid #14b8a6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        cursor: pointer;
        position: relative;
    }
    
    .timeline-item::before {
        content: '📍';
        position: absolute;
        left: -15px;
        top: 50%;
        transform: translateY(-50%) scale(0);
        font-size: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .timeline-item:hover::before {
        transform: translateY(-50%) scale(1);
    }
    
    .timeline-item:hover {
        border-color: #0d7377;
        box-shadow: 0 8px 25px rgba(13, 115, 119, 0.2);
        transform: translateX(10px) scale(1.02);
        border-left-width: 8px;
    }
    
    .timeline-item:active {
        transform: translateX(5px) scale(0.98);
    }
    
    .timeline-item h4 {
        color: #0d7377;
        margin: 0 0 0.75rem 0;
        font-size: 1.05rem;
        font-weight: 700;
    }
    
    .timeline-date {
        color: #ef4444;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .timeline-description {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Recommendations */
    .recommendation-card {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-left: 5px solid;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::after {
        content: '';
        position: absolute;
        right: -50px;
        top: 50%;
        transform: translateY(-50%);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: currentColor;
        opacity: 0.1;
        transition: all 0.4s ease;
    }
    
    .recommendation-card:hover::after {
        right: 20px;
        width: 50px;
        height: 50px;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        transform: translateX(15px) scale(1.02);
        border-left-width: 8px;
    }
    
    .recommendation-card:active {
        transform: translateX(10px) scale(0.98);
    }
    
    .rec-immediate { 
        border-left-color: #ef4444;
    }
    
    .rec-diagnostic { 
        border-left-color: #f59e0b;
    }
    
    .rec-followup { 
        border-left-color: #8b5cf6;
    }
    
    .rec-specialist { 
        border-left-color: #0d7377;
    }
    
    .recommendation-card h4 {
        color: #111827;
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .recommendation-card ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .recommendation-card li {
        margin: 0.75rem 0;
        line-height: 1.7;
        color: #4b5563;
    }
    
    /* Section headers */
    .section-title {
        color: #0d7377;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #14b8a6;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 25px;
        color: #ffffff;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    
    .status-badge::before {
        content: '';
        width: 10px;
        height: 10px;
        background: #ffffff;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { 
            opacity: 1; 
            transform: scale(1);
        }
        50% { 
            opacity: 0.5; 
            transform: scale(1.3);
        }
    }
    
    .status-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5);
    }
    
    .status-badge:active {
        transform: scale(1.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0d7377 0%, #14b8a6 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(13, 115, 119, 0.3);
        transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:active::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 8px 25px rgba(13, 115, 119, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0d7377, #14b8a6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d7377, #0a5a5d);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Alerts */
    .alert-box {
        border: 2px solid;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .alert-success { 
        background: #ecfdf5;
        border-color: #10b981; 
        color: #065f46; 
    }
    
    .alert-warning { 
        background: #fffbeb;
        border-color: #f59e0b; 
        color: #92400e; 
    }
    
    .alert-error { 
        background: #fef2f2;
        border-color: #ef4444; 
        color: #991b1b; 
    }
    
    .alert-info { 
        background: #eff6ff;
        border-color: #3b82f6; 
        color: #1e40af; 
    }
    
    /* View badge */
    .view-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        background: #ffffff;
        border: 2px solid #14b8a6;
        border-radius: 20px;
        color: #0d7377;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(13, 115, 119, 0.1);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
        overflow: hidden;
    }
    
    .view-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(20, 184, 166, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .view-badge:hover::before {
        left: 100%;
    }
    
    .view-badge:hover {
        transform: scale(1.1) rotate(-2deg);
        box-shadow: 0 8px 25px rgba(13, 115, 119, 0.3);
        border-color: #0d7377;
    }
    
    .view-badge:active {
        transform: scale(1.05);
    }
    
    /* Loading */
    .loading-text {
        text-align: center;
        padding: 2rem;
        color: #0d7377;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Image container */
    .image-container {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
    }
    
    .image-container::after {
        content: '🔍';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 1.5rem;
        opacity: 0;
        transform: scale(0) rotate(-180deg);
        transition: all 0.4s ease;
    }
    
    .image-container:hover::after {
        opacity: 1;
        transform: scale(1) rotate(0deg);
    }
    
    .image-container:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 35px rgba(13, 115, 119, 0.2);
        border-color: #14b8a6;
    }
    
    .image-container:active {
        transform: scale(1.02);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f3f4f6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0d7377, #14b8a6);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #14b8a6, #0d7377);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_view_type_from_filename(filename):
    """Extract view type from BRISC filename"""
    filename_lower = filename.lower()
    if '_ax_' in filename_lower:
        return 'AXIAL', '🔵', 'axial'
    elif '_co_' in filename_lower or '_cor_' in filename_lower:
        return 'CORONAL', '🟢', 'coronal'
    elif '_sa_' in filename_lower or '_sag_' in filename_lower:
        return 'SAGITTAL', '🟡', 'sagittal'
    else:
        return 'UNKNOWN', '⚪', 'unknown'

# ============================================================================
# INITIALIZE SYSTEMS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_systems():
    """Load diagnosis and severity assessment systems"""
    cls_path = 'results/classification_new/best_model.pth'
    seg_path = 'best_segmentation.pth'
    
    if not os.path.exists(cls_path) or not os.path.exists(seg_path):
        return None, None, "Model files not found"
    
    try:
        diagnosis_system = BrainTumorDiagnosisSystem(cls_path, seg_path)
        severity_system = TumorSeverityAssessment()
        return diagnosis_system, severity_system, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero-header">
    <h1>🧠 Encephalon Neoplasm Analysis</h1>
    <p>Advanced AI-Powered Brain Tumor Detection & Severity Assessment</p>
    <p style="font-size: 0.95rem; margin-top: 0.5rem;">
        Deep Learning Classification • Tumor Segmentation • Clinical Consultation
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="status-badge" style="width: 100%; justify-content: center;">SYSTEM ACTIVE</div>
        <div style="margin-top: 1.5rem; font-size: 0.9rem; line-height: 2;">
            ✓ Classification Engine Ready<br>
            ✓ Segmentation Module Active<br>
            ✓ Severity Assessment Online<br>
            ✓ Clinical Protocols Loaded
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <h4 style="margin-bottom: 1rem;">⚡ Performance</h4>
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Classification</span>
                <span style="font-weight: 700;">98.10%</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); height: 6px; border-radius: 3px;">
                <div style="background: #ffffff; width: 98.1%; height: 100%; border-radius: 3px;"></div>
            </div>
        </div>
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Segmentation</span>
                <span style="font-weight: 700;">79.72%</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); height: 6px; border-radius: 3px;">
                <div style="background: #ffffff; width: 79.7%; height: 100%; border-radius: 3px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <h4 style="margin-bottom: 1rem;">🎯 Tumor Types</h4>
        <div style="margin: 0.75rem 0; font-size: 0.9rem; line-height: 2;">
            <span style="color: #ef4444;">●</span> Glioma<br>
            <span style="color: #f59e0b;">●</span> Meningioma<br>
            <span style="color: #06b6d4;">●</span> Pituitary<br>
            <span style="color: #10b981;">●</span> No Tumor
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Load systems
with st.spinner("⚡ Initializing AI systems..."):
    diagnosis_system, severity_system, error = load_systems()

if error:
    st.markdown(f"""
    <div class="alert-box alert-error">
        <strong>❌ System Initialization Failed</strong><br>
        {error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div class="alert-box alert-success">
    <strong>✓ All Systems Operational</strong><br>
    Neural networks loaded successfully. Ready for analysis.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# UPLOAD SECTION
# ============================================================================

st.markdown('<div class="section-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select MRI image file",
    type=['jpg', 'jpeg', 'png'],
    help="Supported views: Axial, Coronal, Sagittal",
    label_visibility="collapsed"
)

if uploaded_file:
    view_name, view_icon, view_class = get_view_type_from_filename(uploaded_file.name)
    st.markdown(f"""
    <div class="view-badge">
        {view_icon} {view_name} VIEW
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DIAGNOSIS SECTION
# ============================================================================

if uploaded_file:
    st.markdown('<div class="section-title">🔬 Analysis Results</div>', unsafe_allow_html=True)
    
    # Save file
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    

    # ✅ ADD VALIDATION HERE - BEFORE PROCESSING
    try:
        # Validate if it's a brain MRI
        from mri_validator import MRIValidator
        
        is_valid, validation_msg, metadata = MRIValidator.validate_mri(temp_path, strict=True)
        
        if not is_valid:
            # Show error if not a valid brain MRI
            st.markdown(f"""
            <div class="alert-box alert-error">
                <strong>⚠️ Invalid Brain MRI Image</strong><br><br>
                {validation_msg}<br><br>
                <strong>Requirements:</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                    <li>Must be a brain MRI scan (not a regular photo)</li>
                    <li>Grayscale or low-saturation medical imaging</li>
                    <li>Contains brain structure with characteristic features</li>
                    <li>Size: 50x50 to 4096x4096 pixels</li>
                </ul>
                <br>
                <strong>MRI Validation Score:</strong> {metadata.get('mri_score', 0):.1f}% (Minimum: 50%)<br><br>
                <strong>Details:</strong> {metadata.get('mri_details', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            # Show the uploaded image for reference
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Uploaded Image (Invalid)")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="This does not appear to be a brain MRI scan")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            st.stop()  # Stop execution here
        
        else:
            # Show validation success
            st.markdown(f"""
            <div class="alert-box alert-success">
                <strong>✓ Valid Brain MRI Detected</strong><br>
                {validation_msg}
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="alert-box alert-warning">
            <strong>⚠️ Validation Warning</strong><br>
            Could not fully validate image: {str(e)}<br>
            Proceeding with diagnosis...
        </div>
        """, unsafe_allow_html=True) 
     
    # Layout
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Input MRI Image")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col2:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.markdown('<div class="loading-text">Processing image...</div>', unsafe_allow_html=True)
            progress_bar.progress(20)
            
            import time
            time.sleep(0.3)
            
            status_text.markdown('<div class="loading-text">Running AI classification...</div>', unsafe_allow_html=True)
            progress_bar.progress(50)
            
            time.sleep(0.3)
            
            status_text.markdown('<div class="loading-text">Generating segmentation...</div>', unsafe_allow_html=True)
            progress_bar.progress(80)
            
            results = diagnosis_system.complete_diagnosis(temp_path)
            
            progress_bar.progress(100)
            status_text.markdown('<div class="loading-text">✓ Complete</div>', unsafe_allow_html=True)
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Get results
            classification = results['classification']
            diagnosis = classification['predicted_class']
            confidence = classification['confidence']
            seg_metrics = results['segmentation']['metrics']
            
            # Severity assessment
            severity = severity_system.assess_severity(
                tumor_type=diagnosis,
                segmentation_metrics=seg_metrics,
                classification_confidence=confidence
            )
            
            # Determine severity styling
            risk = severity.get('risk_level','Unknown')
            if risk == 'Low':
                sev_class = 'severity-low'
                color = '#10b981'
            elif risk == 'Moderate':
                sev_class = 'severity-moderate'
                color = '#f59e0b'
            elif risk == 'High':
                sev_class = 'severity-high'
                color = '#ef4444'
            else:
                sev_class = 'severity-critical'
                color = '#dc2626'
            
            st.markdown(f"""
            <div class="severity-box {sev_class}">
                <h2 style="color: {color};">{diagnosis}</h2>
                <div class="severity-score" style="color: {color};">{severity.get('severity_score',0):.0f}</div>
                <div class="severity-grade" style="color: {color};">{severity.get('severity_grade','N/A')}</div>
                <div class="severity-confidence">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{severity.get('risk_level','None')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Tumor Size</div>
                    <div class="metric-value">{seg_metrics.get('area_percentage', 0):.1f}%</div>
                    <div class="metric-subtitle">{severity.get('size_category','N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">View Type</div>
                    <div class="metric-value" style="font-size: 2.2rem;">{view_icon}</div>
                    <div class="metric-subtitle">{view_name}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                biopsy = severity.get('requires_biopsy', {'recommended': False, 'urgency': 'N/A','reason': 'N/A'})
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Biopsy</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{'YES' if biopsy['recommended'] else 'NO'}</div>
                    <div class="metric-subtitle">{biopsy['urgency']}</div>
                    
                </div>
                """, unsafe_allow_html=True)
            
            # Timeline
            st.markdown('<div class="section-title">📅 Consultation Timeline</div>', unsafe_allow_html=True)
            
            timeline = severity.get('consultation_timeline', {
                'neurosurgeon': 'N/A',
                'imaging': 'N/A',
                'biopsy': 'N/A'
            })
            
            col_t1, col_t2, col_t3 = st.columns(3)
            
            with col_t1:
                st.markdown(f"""
                <div class="timeline-item">
                    <h4>🏥 Neurosurgeon</h4>
                    <div class="timeline-date">{timeline.get('neurosurgeon','N/A')}</div>
                    <div class="timeline-description">Primary Consultation</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_t2:
                st.markdown(f"""
                <div class="timeline-item">
                    <h4>📊 Follow-up Imaging</h4>
                    <div class="timeline-date">{timeline.get('imaging','N/A')}</div>
                    <div class="timeline-description">MRI Re-assessment</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_t3:
                st.markdown(f"""
                <div class="timeline-item">
                    <h4>🔬 Biopsy</h4>
                    <div class="timeline-date">{timeline.get('biopsy','N/A')}</div>
                    <div class="timeline-description">If Required</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown('<div class="section-title">💊 Clinical Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = severity.get('clinical_recommendations',{
                'immediate_actions': ['Routine follow-up'],
                'diagnostic_tests': ['standard checkup'],
                'specialist_consultation': ['General physician'],
                'imaging_followup': ['Annual screening']
            })
            
            col_r1, col_r2 = st.columns(2, gap="large")
            
            with col_r1:
                st.markdown("""
                <div class="recommendation-card rec-immediate">
                    <h4>🚨 Immediate Actions</h4>
                    <ul>
                """, unsafe_allow_html=True)
                for action in recommendations.get('immediate_actions',['Routine follow-up']):
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="recommendation-card rec-diagnostic">
                    <h4>🔬 Diagnostic Tests</h4>
                    <ul>
                """, unsafe_allow_html=True)
                for test in recommendations['diagnostic_tests'][:5]:
                    st.markdown(f"<li>{test}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col_r2:
                st.markdown("""
                <div class="recommendation-card rec-specialist">
                    <h4>👨‍⚕️ Specialists</h4>
                    <ul>
                """, unsafe_allow_html=True)
                for specialist in recommendations['specialist_consultation']:
                    st.markdown(f"<li>{specialist}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="recommendation-card rec-followup">
                    <h4>📅 Imaging Schedule</h4>
                    <ul>
                """, unsafe_allow_html=True)
                for schedule in recommendations['imaging_followup']:
                    st.markdown(f"<li>{schedule}</li>", unsafe_allow_html=True)
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Segmentation
            st.markdown('<div class="section-title">🗺️ Tumor Segmentation</div>', unsafe_allow_html=True)
            
            if diagnosis == 'No Tumor':
                st.markdown("""
                <div class="alert-box alert-success">
                    <strong>✓ No Tumor Detected</strong><br>
                    Classification confirmed healthy brain tissue. No abnormalities identified.
                </div>
                """, unsafe_allow_html=True)
            
            elif diagnosis == 'Pituitary':
                st.markdown(f"""
                <div class="alert-box alert-info">
                    <strong>ℹ️ Pituitary Tumor Detected</strong><br><br>
                    <strong>Classification Confidence:</strong> {confidence:.1f}%<br>
                    <strong>Size Assessment:</strong> {severity['size_category']}<br><br>
                    <strong>Note:</strong> Segmentation optimized for glioma/meningioma. 
                    Pituitary diagnosis relies on high-accuracy classification.<br><br>
                    <strong>Recommendation:</strong> Multi-view MRI with contrast + endocrine evaluation.
                </div>
                """, unsafe_allow_html=True)
            
            elif seg_metrics.get('tumor_detected'):
                col_seg1, col_seg2 = st.columns([1, 1.5], gap="large")
                
                with col_seg1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Quantitative Analysis")
                    
                    area_pct = seg_metrics['area_percentage']
                    area_px = seg_metrics['area_pixels']
                    
                    st.markdown(f"""
                    <div style="margin: 1.75rem 0;">
                        <div style="color: #6b7280; font-size: 0.85rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">Tumor Coverage</div>
                        <div style="color: #0d7377; font-size: 3rem; font-weight: 900;">
                            {area_pct:.2f}%
                        </div>
                        <div style="color: #9ca3af; font-size: 0.95rem; margin-top: 0.5rem;">{area_px:,} pixels</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'centroid' in seg_metrics:
                        cent = seg_metrics['centroid']
                        st.markdown(f"""
                        <div style="margin: 1.75rem 0; padding: 1.25rem; background: #f9fafb; border: 2px solid #e5e7eb; border-radius: 12px;">
                            <div style="color: #6b7280; font-size: 0.85rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">Tumor Centroid</div>
                            <div style="color: #0d7377; font-weight: 700; font-size: 1.3rem;">
                                X: {cent['x']} | Y: {cent['y']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_seg2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 🗺️ Segmentation Map")
                    
                    report_path = 'temp_segmentation_viz.png'
                    diagnosis_system.generate_diagnostic_report(results, save_path=report_path)
                    
                    
                    if os.path.exists(report_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(report_path, use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        os.remove(report_path)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class="alert-box alert-warning">
                    <strong>⚠️ Classification-Segmentation Mismatch</strong><br><br>
                    <strong>Classification:</strong> {diagnosis} ({confidence:.1f}% confidence)<br>
                    <strong>Segmentation:</strong> No clear boundaries detected<br><br>
                    <strong>Possible Causes:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        <li>Very small/diffuse tumor (< 0.2% area)</li>
                        <li>Low contrast boundaries in this view</li>
                        <li>Different tumor characteristics</li>
                    </ul>
                    <strong>Recommendation:</strong> Upload multiple views (Axial + Coronal + Sagittal).
                </div>
                """, unsafe_allow_html=True)
            
            # Download Reports
            st.markdown('<div class="section-title">📥 Download Reports</div>', unsafe_allow_html=True)
            
            final_report_path = f'diagnosis_report_{uploaded_file.name}.png'
            diagnosis_system.generate_diagnostic_report(results, save_path=final_report_path)

            # Generate severity assessment visualization
            severity_viz_path = f'severity_assessment_{uploaded_file.name}.png'
            severity_system.generate_severity_visualization(severity, save_path=severity_viz_path)
            
            # Display both reports
            col_rep1, col_rep2 = st.columns(2, gap="large")
            with col_rep1:
                st.markdown("#### 🔬 Diagnostic Report")
                if os.path.exists(final_report_path):
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(final_report_path, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col_rep2:
                st.markdown("#### 📊 Severity Assessment")
                if os.path.exists(severity_viz_path):
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(severity_viz_path, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if os.path.exists(final_report_path):
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(final_report_path, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_dl1, col_dl2, col_dl3 ,col_dl4 = st.columns(4)
                
                with col_dl1:
                    with open(final_report_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Visual Report (PNG)",
                            data=file,
                            file_name=final_report_path,
                            mime="image/png",
                            use_container_width=True
                        )
                
                with col_dl2:
                    with open(severity_viz_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Severity Visualization (PNG)",
                            data=file,
                            file_name=severity_viz_path,
                            mime="image/png",
                            use_container_width=True
                        )
                with col_dl3:
                    json_data = {
                        'patient_info': {
                            'filename': uploaded_file.name,
                            'scan_date': datetime.now().strftime('%Y-%m-%d'),
                            'view_type': view_name
                        },
                        'diagnosis': {
                            'tumor_type': diagnosis,
                            'confidence': confidence,
                            'all_probabilities': classification['all_probabilities']
                        },
                        'severity_assessment': {
                            'grade': severity['severity_grade'],
                            'score': severity['severity_score'],
                            'risk_level': severity['risk_level'],
                            'urgency': severity['urgency'],
                            'size_category': severity['size_category']
                        },
                        'segmentation': seg_metrics,
                        'consultation_timeline': timeline,
                        'recommendations': {
                            'immediate_actions': recommendations['immediate_actions'],
                            'specialists': recommendations['specialist_consultation'],
                            'imaging_schedule': recommendations['imaging_followup']
                        },
                        'biopsy': severity.get('requires_biopsy', 'N/A'),
                        'timestamp': results['timestamp']
                    }
                    
                    st.download_button(
                        label="⬇️ Clinical Data (JSON)",
                        data=json.dumps(json_data, indent=2),
                        file_name=f'clinical_data_{uploaded_file.name}.json',
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col_dl4:
                    text_report = severity_system.generate_severity_report(severity)
                    
                    st.download_button(
                        label="⬇️ Text Report (TXT)",
                        data=text_report,
                        file_name=f'severity_report_{uploaded_file.name}.txt',
                        mime="text/plain",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.markdown(f"""
            <div class="alert-box alert-error">
                <strong>❌ Error During Analysis</strong><br>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)
            import traceback
            with st.expander("Show technical details"):
                st.code(traceback.format_exc())
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    # Welcome Screen
    st.markdown('<div class="section-title">📊 System Overview</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4, gap="large")
    
    with col_f1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Classification</div>
            <div class="metric-value">98.10%</div>
            <div class="metric-subtitle">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Segmentation</div>
            <div class="metric-value">79.72%</div>
            <div class="metric-subtitle">Dice Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Process Time</div>
            <div class="metric-value">~3s</div>
            <div class="metric-subtitle">Real-time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Detection</div>
            <div class="metric-value">4</div>
            <div class="metric-subtitle">Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">✨ Key Features</div>', unsafe_allow_html=True)
    
    col_feat1, col_feat2 = st.columns(2, gap="large")
    
    with col_feat1:
        st.markdown("""
        <div class="card">
            <h3>🎯 AI-Powered Diagnosis</h3>
            <div style="color: #4b5563; line-height: 1.8; margin-top: 1rem; font-size: 0.95rem;">
                ✓ Deep learning classification (98.10% accuracy)<br>
                ✓ Precise tumor segmentation (79.72% Dice)<br>
                ✓ Multi-view orientation support<br>
                ✓ Real-time processing<br>
                ✓ Automated quality checks
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📊 Severity Assessment</h3>
            <div style="color: #4b5563; line-height: 1.8; margin-top: 1rem; font-size: 0.95rem;">
                ✓ WHO-adapted grading system<br>
                ✓ Multi-factor severity scoring<br>
                ✓ Risk stratification<br>
                ✓ Size-based categorization<br>
                ✓ Geometric feature analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown("""
        <div class="card">
            <h3>🏥 Clinical Consultation</h3>
            <div style="color: #4b5563; line-height: 1.8; margin-top: 1rem; font-size: 0.95rem;">
                ✓ Automated timeline calculation<br>
                ✓ Specialist referral routing<br>
                ✓ Evidence-based protocols<br>
                ✓ Treatment pathway guidance<br>
                ✓ Biopsy recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📋 Comprehensive Reporting</h3>
            <div style="color: #4b5563; line-height: 1.8; margin-top: 1rem; font-size: 0.95rem;">
                ✓ Visual diagnostic reports (PNG)<br>
                ✓ Structured clinical data (JSON)<br>
                ✓ Severity assessment docs (TXT)<br>
                ✓ Multi-format downloads<br>
                ✓ EHR-compatible output
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="card" style="text-align: center;">
    <h3 style="color: #0d7377; margin-bottom: 1.5rem;">⚠️ Medical Disclaimer</h3>
    <p style="color: #4b5563; line-height: 1.8; margin: 1rem 0; font-size: 0.95rem;">
    This AI-powered diagnostic system is designed for educational and research purposes only.<br>
    All diagnoses, severity assessments, and clinical recommendations must be validated by qualified medical professionals.<br>
    This system is not a substitute for professional medical judgment or comprehensive patient evaluation.<br>
    Do not make treatment decisions based solely on AI analysis without proper medical consultation.
    </p>
    <div style="margin-top: 2rem; padding-top: 2rem; border-top: 2px solid #e5e7eb;">
        <div style="color: #0d7377; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">
            Encephalon AI Neurodiagnostic Platform v2.0
        </div>
        <div style="color: #6b7280; font-size: 0.85rem;">
            ResNet18 Architecture • 98.10% Accuracy • Streamlit Interface • © 2025
        </div>
    </div>
</div>
""", unsafe_allow_html=True)