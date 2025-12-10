"""
Batch Processing Feature - Add this as a new page in your Streamlit app
Place this in a new file: pages/2_Batch_Processing.py
"""

import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
import zipfile
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

try:
    from complete_diagnosis import BrainTumorDiagnosisSystem
except ImportError:
    st.error("❌ Could not import diagnosis system")
    st.stop()

# Page config
st.set_page_config(page_title="Batch Processing", page_icon="📊", layout="wide")

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #00897B 0%, #004D40 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>📊 Batch Processing</h1>
    <p style='color: #B2DFDB; margin: 0.5rem 0 0 0;'>Process multiple MRI scans simultaneously</p>
</div>
""", unsafe_allow_html=True)

# Load system
@st.cache_resource
def load_system():
    cls_path = 'results/classification_new/best_model.pth'
    seg_path = 'best_segmentation.pth'
    return BrainTumorDiagnosisSystem(cls_path, seg_path)

system = load_system()

# Upload multiple files
st.markdown("### 📤 Upload Multiple MRI Scans")
uploaded_files = st.file_uploader(
    "Select multiple images",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Upload multiple MRI scans for batch processing"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    # Processing options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        generate_reports = st.checkbox("Generate individual reports", value=True)
        generate_summary = st.checkbox("Generate summary CSV", value=True)
    
    with col2:
        if st.button("🚀 Process All", type="primary", use_container_width=True):
            
            # Create results storage
            results_list = []
            report_files = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(f"**Processing {idx+1}/{len(uploaded_files)}:** {uploaded_file.name}")
                
                # Save temp file
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Run diagnosis
                    result = system.complete_diagnosis(temp_path)
                    
                    # Extract key metrics
                    classification = result['classification']
                    seg_metrics = result['segmentation']['metrics']
                    
                    # Store results
                    results_list.append({
                        'Filename': uploaded_file.name,
                        'Diagnosis': classification['predicted_class'],
                        'Confidence': f"{classification['confidence']:.1f}%",
                        'Tumor_Detected': classification['has_tumor'],
                        'Tumor_Area_%': seg_metrics.get('area_percentage', 0),
                        'Centroid_X': seg_metrics.get('centroid', {}).get('x', 0),
                        'Centroid_Y': seg_metrics.get('centroid', {}).get('y', 0),
                        'Timestamp': result['timestamp']
                    })
                    
                    # Generate report if requested
                    if generate_reports:
                        report_path = f"batch_report_{uploaded_file.name}.png"
                        system.generate_diagnostic_report(result, save_path=report_path)
                        report_files.append(report_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    results_list.append({
                        'Filename': uploaded_file.name,
                        'Diagnosis': 'ERROR',
                        'Confidence': '0%',
                        'Error': str(e)
                    })
                
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Batch Processing Results")
            
            # Create DataFrame
            df = pd.DataFrame(results_list)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(df)
            tumors_detected = df['Tumor_Detected'].sum() if 'Tumor_Detected' in df.columns else 0
            avg_confidence = df['Confidence'].str.rstrip('%').astype(float).mean() if 'Confidence' in df.columns else 0
            
            col1.metric("Total Scans", total)
            col2.metric("Tumors Detected", tumors_detected)
            col3.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            col4.metric("Success Rate", f"{(total - df['Diagnosis'].eq('ERROR').sum()) / total * 100:.0f}%")
            
            # Display full results table
            st.dataframe(df, use_container_width=True)
            
            # Distribution chart
            if 'Diagnosis' in df.columns:
                st.markdown("### 📈 Diagnosis Distribution")
                diagnosis_counts = df['Diagnosis'].value_counts()
                st.bar_chart(diagnosis_counts)
            
            # Download options
            st.markdown("---")
            st.markdown("### ⬇️ Download Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            # CSV download
            with col_dl1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Summary CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # ZIP all reports
            with col_dl2:
                if report_files:
                    # Create ZIP file in memory
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for report_file in report_files:
                            if os.path.exists(report_file):
                                zip_file.write(report_file, os.path.basename(report_file))
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download All Reports (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"diagnostic_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    # Cleanup report files
                    for report_file in report_files:
                        if os.path.exists(report_file):
                            os.remove(report_file)

else:
    # Instructions when no files uploaded
    st.info("👆 Upload multiple MRI scans to begin batch processing")
    
    st.markdown("### ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Fast Processing**
        - Process multiple scans
        - Automatic view detection
        - Parallel-ready design
        """)
    
    with col2:
        st.markdown("""
        **📊 Comprehensive Results**
        - Summary statistics
        - CSV export
        - Distribution charts
        """)
    
    with col3:
        st.markdown("""
        **📦 Easy Export**
        - Download all reports
        - ZIP archive
        - CSV data export
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
💡 <strong>Tip:</strong> For best results, ensure all images are in the same format and quality
</div>
""", unsafe_allow_html=True)