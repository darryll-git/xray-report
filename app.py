import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.preprocessing import XRayPreprocessor
from utils.visualization import overlay_mask, draw_measurements, draw_rib_markers, create_comparison_figure
from models.segmentation import AnatomySegmenter
from models.classification import XRayClassifier
from models.measurements import XRayMeasurements
from models.report_generator import ReportGenerator
from models.pulmonary_analysis import PulmonaryEdemaAnalyzer, LungFissureDetector
from utils.visualization import create_enhanced_visualization

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide"
)

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results = None
    st.session_state.image = None
    st.session_state.models_loaded = False

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models():
    """Load all models with better error handling - NOT cached to allow retry"""
    
    preprocessor = XRayPreprocessor()
    segmenter = AnatomySegmenter(device=device)
    classifier = XRayClassifier(device=device)
    measurements = XRayMeasurements()
    
    # Track loading status
    status = {
        'view_classifier': False,
        'abnormality_detector': False,
        'report_generator': False
    }
    
    # Load view classifier (CRITICAL - must succeed)
    with st.spinner("Loading view classifier... (This is required)"):
        try:
            success = classifier.load_view_classifier()
            status['view_classifier'] = success
            if success:
                st.success("✓ View classifier loaded successfully")
        except Exception as e:
            st.error(f"❌ CRITICAL: Failed to load view classifier")
            st.error(f"Error: {str(e)}")
            st.error("This model is required for analysis. Please:")
            st.info("1. Check your internet connection\n2. Wait a moment and refresh the page\n3. If problem persists, the model server may be down")
            raise

    # Load abnormality detector (IMPORTANT but not critical)
    with st.spinner("Loading abnormality detector..."):
        try:
            success = classifier.load_abnormality_detector()
            status['abnormality_detector'] = success
            if success:
                st.success("✓ Abnormality detector loaded")
            else:
                st.warning("⚠ Abnormality detector failed - will use basic analysis")
        except Exception as e:
            st.warning(f"⚠ Could not load abnormality detector: {e}")
            st.info("Analysis will continue with limited abnormality detection")
    
    # Initialize report generator
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            report_gen = ReportGenerator(api_key=groq_api_key)
            status['report_generator'] = True
            st.success("✓ Report generator initialized")
        else:
            report_gen = ReportGenerator(api_key=None)
            status['report_generator'] = False
            st.warning("⚠ Groq API key not found - using template reports")
    except Exception as e:
        st.warning(f"⚠ Report generator issue: {e}")
        report_gen = None
    
    return preprocessor, segmenter, classifier, measurements, report_gen, status


def create_visualization(image, results):
    """Use the enhanced visualization"""
    return create_enhanced_visualization(image, results)


def analyze_xray(image, preprocessor, segmenter, classifier, measurements, report_gen, patient_info=None):
    """Complete X-ray analysis pipeline with pulmonary analysis"""
    
    results = {}
    
    # Convert to numpy for processing
    img_array = np.array(image.convert('L'))
    
    with st.spinner("Step 1/7: Preprocessing image..."):
        enhanced_img = preprocessor.enhance_contrast(image)
        tensor_xrv = preprocessor.preprocess_for_torchxrayvision(enhanced_img)
    
    with st.spinner("Step 2/7: Classifying view and segmentation..."):
        try:
            view_result = classifier.classify_view(image, tensor_xrv)
            results['view_classification'] = view_result
            ml_mask = view_result.get('mask', None)

            if view_result and 'age' in view_result:
                results['patient_age'] = view_result['age']
                results['patient_female_prob'] = 1.0 if view_result.get('gender') == 'Female' else 0.0

            rotation_result = classifier.check_rotation(image)
            rotation_result['assessment'] = rotation_result.get('message', "Rotation assessment unavailable")
            rotation_result['symmetric'] = rotation_result.get('symmetric', False)
            results['rotation'] = rotation_result
        except Exception as e:
            st.error(f"Error in view classification: {e}")
            raise
    
    with st.spinner("Step 3/7: Segmenting anatomical structures..."):
        masks = segmenter.segment_all(enhanced_img, use_ml_model=True, ml_masks=ml_mask)
        results['masks'] = masks
        results['segmentation_method'] = masks.get('method', 'unknown')
    
    with st.spinner("Step 4/7: Detecting abnormalities..."):
        try:
            if classifier.torchxrv_model is not None:
                abnormality_results = classifier.detect_abnormalities(tensor_xrv)
                findings = abnormality_results.get('findings', {})
                results['abnormalities'] = findings
                results['significant_findings'] = abnormality_results.get('significant', {})
            else:
                st.warning("Skipping abnormality detection - model not loaded")
                results['abnormalities'] = {}
                results['significant_findings'] = {}
                findings = {}
        except Exception as e:
            st.warning(f"Error in abnormality detection: {e}")
            results['abnormalities'] = {}
            results['significant_findings'] = {}
            findings = {}
    
    # PULMONARY ANALYSIS
    with st.spinner("Step 5/7: Analyzing pulmonary patterns..."):
        try:
            edema_analyzer = PulmonaryEdemaAnalyzer()
            fissure_detector = LungFissureDetector()
            
            img_gray = np.array(image.convert('L'))
            
            # Edema pattern analysis
            edema_pattern = edema_analyzer.analyze_edema_pattern(
                img_gray, 
                masks['lungs'],
                findings
            )
            results['edema_pattern'] = edema_pattern
            
            # Fissure detection
            fissures = fissure_detector.detect_fissures(img_gray, masks['lungs'])
            results['fissures'] = fissures
            
        except Exception as e:
            st.error(f"Error in pulmonary analysis: {e}")
            results['edema_pattern'] = None
            results['fissures'] = None
    
    with st.spinner("Step 6/7: Measuring technical parameters..."):
        rib_result = measurements.count_posterior_ribs(img_array, masks['lungs'])
        results['inspiration'] = rib_result
        
        penetration_result = measurements.check_penetration(img_array)
        results['penetration'] = penetration_result
        
        trachea_result = measurements.check_trachea_position(img_array)
        results['trachea'] = trachea_result
    
    with st.spinner("Step 7/7: Calculating CTR and generating report..."):
        if ml_mask is not None:
            ctr_result = segmenter.calculate_ctr_from_mask(ml_mask)
        else:
            ctr_result = measurements.calculate_cardiothoracic_ratio(masks['heart'], masks['lungs'])
            
        if ctr_result and isinstance(ctr_result, dict):
            ctr_value = ctr_result.get('ctr')
            if ctr_value is not None:
                if 'interpretation' in ctr_result:
                    ctr_result['assessment'] = ctr_result['interpretation']
                else:
                    if ctr_value > 0.55:
                        assessment = f"Marked cardiomegaly - CTR {ctr_value:.2f} (>0.55)"
                    elif ctr_value > 0.50:
                        assessment = f"Borderline cardiomegaly - CTR {ctr_value:.2f} (>0.50)"
                    else:
                        assessment = f"Normal cardiac size - CTR {ctr_value:.2f} (≤0.50)"
                    ctr_result['assessment'] = assessment
            
        results['ctr'] = ctr_result
        
        angles_result = measurements.check_costophrenic_angles(masks['lungs'])
        results['costophrenic_angles'] = angles_result
        
        if report_gen:
            try:
                report = report_gen.generate_structured_report(image, results, patient_info)
                results['report'] = report
            except Exception as e:
                st.warning(f"Error generating report: {e}")
                results['report'] = "Report generation failed. Please review findings manually."
        else:
            results['report'] = "Report generator not available. Please review findings manually."
        
        if patient_info:
            results['patient_info'] = patient_info
    
    return results


def display_results(image, results):
    """Display analysis results - keeping existing implementation"""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Report", 
        "🖼️ Visualizations",
        "🔧 Technical", 
        "🫁 Anatomy", 
        "🫀 Pulmonary Patterns",
        "⚕️ Findings"
    ])
    
    with tab1:
        st.subheader("Comprehensive Radiology Report")
        st.text_area("Report", results['report'], height=600)
        st.download_button(
            label="Download Report",
            data=results['report'],
            file_name="chest_xray_report.txt",
            mime="text/plain"
        )
    
    with tab2:
        st.subheader("Anatomical Segmentation & Measurements")
        
        viz_image = create_visualization(image, results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original X-Ray", use_container_width=True)
        with col2:
            st.image(viz_image, caption="Annotated Analysis", use_container_width=True)
        
        st.markdown("""
        **Color Legend:**
        - 🔵 Blue: Lung fields
        - 🔴 Red: Cardiac silhouette  
        - 🟡 Yellow: Posterior ribs (numbered)
        - 🟢 Green: Sharp costophrenic angles
        - 🟠 Orange: Blunted costophrenic angles
        - 🔷 Light Blue (dashed): Major fissures
        - 🟣 Purple (dashed): Minor fissure
        """)
    
    with tab3:
        st.subheader("Technical Considerations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Projection")
            view = results['view_classification']
            st.info(f"**View:** {view['view']}")
            st.caption(f"Confidence: {view['confidence']:.2%}")
            
            if 'patient_age' in results:
                st.markdown("#### Patient Information")
                st.metric("Estimated Age", f"{results['patient_age']:.0f} years")
            
            st.markdown("#### Rotation")
            rotation = results.get('rotation', {})
            if rotation.get('symmetric', False):
                st.success(rotation.get('assessment', 'Normal'))
            else:
                st.warning(rotation.get('assessment', 'Check rotation'))
            
        with col2:
            st.markdown("#### Inspiration")
            insp = results.get('inspiration', {})
            rib_count = insp.get('count', 0)
            st.metric("Posterior Ribs Visible", rib_count)
            
            st.markdown("#### Penetration")
            pen = results.get('penetration', {})
            if pen.get('adequate', False):
                st.success(pen.get('assessment', 'Adequate'))
            else:
                st.warning(pen.get('assessment', 'Check penetration'))
    
    with tab4:
        st.subheader("Anatomical Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Airway")
            trachea = results.get('trachea', {})
            if trachea.get('midline', True):
                st.success(trachea.get('assessment', 'Midline'))
            else:
                st.warning(trachea.get('assessment', 'Deviated'))
        
        with col2:
            st.markdown("#### Cardiac Silhouette")
            ctr = results.get('ctr', {})
            if ctr:
                ctr_value = ctr.get('ctr', 0.0)
                st.metric("CTR", f"{ctr_value:.3f}")
                
                if ctr_value < 0.5:
                    st.success(ctr.get('assessment', 'Normal'))
                else:
                    st.warning(ctr.get('assessment', 'Enlarged'))
    
    with tab5:
        st.subheader("Pulmonary Pattern Analysis")
        
        if 'edema_pattern' in results and results['edema_pattern']:
            pattern = results['edema_pattern']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Edema Assessment")
                pattern_type = pattern.get('pattern_type', 'unknown')
                confidence = pattern.get('confidence', 0)
                
                st.write(f"**Pattern:** {pattern_type}")
                st.write(f"**Confidence:** {confidence:.0%}")
                
                features = pattern.get('supporting_features', [])
                if features:
                    st.markdown("**Features:**")
                    for f in features:
                        st.markdown(f"- {f}")
            
            with col2:
                kerley = pattern.get('kerley_lines')
                if kerley:
                    st.markdown("#### Interstitial Markings")
                    st.write(kerley.get('interpretation', 'None detected'))
        
        if 'fissures' in results and results['fissures']:
            st.divider()
            st.markdown("#### Pulmonary Fissures")
            
            fissures = results['fissures']
            major = fissures.get('major_fissures', [])
            minor = fissures.get('minor_fissure')
            
            if major or minor:
                st.success(f"✓ {len(major)} major fissure(s) detected")
                if minor:
                    st.success("✓ Minor fissure detected")
    
    with tab6:
        st.subheader("Abnormality Detection")
        
        sig_findings = results.get('significant_findings', {})
        
        if sig_findings:
            st.warning(f"⚠️ {len(sig_findings)} significant finding(s)")
            
            for pathology, score in sig_findings.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{pathology}**")
                with col2:
                    if isinstance(score, (float, int)):
                        st.write(f"{score:.1%}")
                        st.progress(float(score))
        else:
            st.success("✓ No significant abnormalities detected")


def main():
    st.title("🫁 AI Chest X-Ray Analyzer")
    st.markdown("Comprehensive automated analysis of chest X-rays")
    
    with st.sidebar:
        st.header("About")
        st.info("""
        This application performs comprehensive chest X-ray analysis including:
        
        - Technical quality assessment
        - Anatomical segmentation
        - Abnormality detection
        - Pulmonary pattern analysis
        - Structured report generation
        
        **Note:** AI-assisted tool. All findings should be verified by a qualified radiologist.
        """)
        
        st.header("System Status")
        if torch.cuda.is_available():
            st.success(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info("Running on CPU")
    
    # Load models with explicit button or auto-load
    if not st.session_state.get('models_loaded', False):
        st.warning("⚠️ Models not yet loaded. Click below to initialize.")
        
        if st.button("🚀 Initialize AI Models", type="primary", use_container_width=True):
            try:
                with st.spinner("Initializing AI models... This may take 1-2 minutes on first run."):
                    preprocessor, segmenter, classifier, measurements, report_gen, status = load_models()
                    
                    # Store in session state
                    st.session_state.preprocessor = preprocessor
                    st.session_state.segmenter = segmenter
                    st.session_state.classifier = classifier
                    st.session_state.measurements = measurements
                    st.session_state.report_gen = report_gen
                    st.session_state.model_status = status
                    st.session_state.models_loaded = True
                    
                    st.success("✓ Models loaded successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.error("Please refresh the page and try again.")
                st.stop()
        
        st.stop()  # Don't show upload until models are loaded
    
    # Show model status
    with st.sidebar:
        st.header("Model Status")
        status = st.session_state.get('model_status', {})
        if status.get('view_classifier'):
            st.success("✓ View Classifier")
        if status.get('abnormality_detector'):
            st.success("✓ Abnormality Detector")
        if status.get('report_generator'):
            st.success("✓ Report Generator")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image", 
        type=['png', 'jpg', 'jpeg', 'dcm'],
        help="Upload a chest X-ray image"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            st.divider()
            
            # Patient info
            st.subheader("Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name", help="Patient's full name")
                mrn = st.text_input("MRN", help="Medical Record Number")
                dob = st.text_input("DOB", help="YYYY-MM-DD")
            with col2:
                referring = st.text_input("Referring Physician")
                gender = st.radio("Gender", ["Male", "Female", "Other"])
                history = st.text_area("Clinical History", height=100)
            
            st.divider()
            
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                patient_info = {
                    'name': name or 'ANONYMOUS',
                    'mrn': mrn or 'NOT PROVIDED',
                    'dob': dob or 'NOT PROVIDED',
                    'gender': gender,
                    'referring_physician': referring or 'NOT PROVIDED',
                    'history': history or 'Routine examination',
                    'exam_date': 'CURRENT'
                }
                
                try:
                    results = analyze_xray(
                        image, 
                        st.session_state.preprocessor,
                        st.session_state.segmenter,
                        st.session_state.classifier,
                        st.session_state.measurements,
                        st.session_state.report_gen,
                        patient_info
                    )
                    
                    st.session_state.results = results
                    st.session_state.image = image
                    st.session_state.analysis_done = True
                    
                    st.success("✓ Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    with st.expander("View error details"):
                        st.code(traceback.format_exc())
            
            if st.session_state.analysis_done and st.session_state.results:
                st.divider()
                display_results(st.session_state.image, st.session_state.results)
                
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")


if __name__ == "__main__":
    main()