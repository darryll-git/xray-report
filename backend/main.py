from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import numpy as np
import io
import base64
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

# Import models and utils
from utils.preprocessing import XRayPreprocessor
from utils.visualization import create_enhanced_visualization
from models.segmentation import AnatomySegmenter
from models.classification import XRayClassifier
from models.measurements import XRayMeasurements
from models.report_generator import ReportGenerator
from models.pulmonary_analysis import PulmonaryEdemaAnalyzer, LungFissureDetector
from models.clinical_rules import create_clinical_summary

# Global state for models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Initializing AI models... This may take a minute.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    preprocessor = XRayPreprocessor()
    segmenter = AnatomySegmenter(device=device)
    classifier = XRayClassifier(device=device)
    measurements = XRayMeasurements()
    
    # Load view classifier
    try:
        classifier.load_view_classifier()
        print("✓ View classifier loaded")
    except Exception as e:
        print(f"❌ Failed to load view classifier: {e}")
        
    # Load abnormality detector
    try:
        classifier.load_abnormality_detector()
        print("✓ Abnormality detector loaded")
    except Exception as e:
        print(f"⚠ Could not load abnormality detector: {e}")
        
    # Initialize report generator
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        report_gen = ReportGenerator(api_key=groq_api_key)
        print("✓ Report generator initialized")
    else:
        report_gen = ReportGenerator(api_key=None)
        print("⚠ Groq API key not found - using template reports")
        
    ml_models['preprocessor'] = preprocessor
    ml_models['segmenter'] = segmenter
    ml_models['classifier'] = classifier
    ml_models['measurements'] = measurements
    ml_models['report_gen'] = report_gen
    
    yield
    
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def get_status():
    status = {
        'view_classifier': ml_models['classifier'].view_classifier is not None if 'classifier' in ml_models else False,
        'abnormality_detector': ml_models['classifier'].torchxrv_model is not None if 'classifier' in ml_models else False,
        'report_generator': ml_models['report_gen'].client is not None if 'report_gen' in ml_models else False,
    }
    return {"status": status}

def image_to_base64(img):
    buffered = io.BytesIO()
    # Check if image is numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(v) for v in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(v) for v in obj]
    return obj

@app.post("/api/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    name: str = Form(default="ANONYMOUS"),
    mrn: str = Form(default="NOT PROVIDED"),
    dob: str = Form(default="NOT PROVIDED"),
    gender: str = Form(default="Other"),
    referring_physician: str = Form(default="NOT PROVIDED"),
    history: str = Form(default="Routine examination")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
    patient_info = {
        'name': name,
        'mrn': mrn,
        'dob': dob,
        'gender': gender,
        'referring_physician': referring_physician,
        'history': history,
        'exam_date': 'CURRENT'
    }
    
    preprocessor = ml_models.get('preprocessor')
    segmenter = ml_models.get('segmenter')
    classifier = ml_models.get('classifier')
    measurements = ml_models.get('measurements')
    report_gen = ml_models.get('report_gen')
    
    if not all([preprocessor, segmenter, classifier, measurements]):
        raise HTTPException(status_code=500, detail="Models not fully initialized")

    results = {}
    img_array = np.array(image.convert('L'))
    
    # 1. Preprocessing
    enhanced_img = preprocessor.enhance_contrast(image)
    tensor_xrv = preprocessor.preprocess_for_torchxrayvision(enhanced_img)
    
    # 2. View and segmentation
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
        print(f"Error in view classification: {e}")
        results['view_classification'] = {'view': 'Unknown', 'confidence': 0.0}
        ml_mask = None
        results['rotation'] = {'assessment': 'Rotation assessment unavailable', 'symmetric': False}
        
    # 3. Segmenting
    masks = segmenter.segment_all(enhanced_img, use_ml_model=True, ml_masks=ml_mask)
    results['masks'] = masks
    results['segmentation_method'] = masks.get('method', 'unknown')
    
    # 4. Detecting abnormalities
    if classifier.torchxrv_model is not None:
        abnormality_results = classifier.detect_abnormalities(tensor_xrv)
        findings = abnormality_results.get('findings', {})
        results['abnormalities'] = findings
        results['significant_findings'] = abnormality_results.get('significant', {})
        results['clinical_summary'] = create_clinical_summary(findings, results.get('view_classification', {}))
    else:
        results['abnormalities'] = {}
        results['significant_findings'] = {}
        results['clinical_summary'] = None
        findings = {}
        
    # 5. Pulmonary Analysis
    try:
        edema_analyzer = PulmonaryEdemaAnalyzer()
        fissure_detector = LungFissureDetector()
        img_gray = np.array(image.convert('L'))
        
        edema_pattern = edema_analyzer.analyze_edema_pattern(img_gray, masks['lungs'], findings)
        results['edema_pattern'] = edema_pattern
        
        fissures = fissure_detector.detect_fissures(img_gray, masks['lungs'])
        results['fissures'] = fissures
    except Exception as e:
        print(f"Error in pulmonary analysis: {e}")
        results['edema_pattern'] = None
        results['fissures'] = None
        
    # 6. Measurements
    rib_result = measurements.count_posterior_ribs(img_array, masks['lungs'])
    results['inspiration'] = rib_result
    
    penetration_result = measurements.check_penetration(img_array)
    results['penetration'] = penetration_result
    
    trachea_result = measurements.check_trachea_position(img_array)
    results['trachea'] = trachea_result
    
    # 7. CTR and Report
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
            print(f"Error generating report: {e}")
            results['report'] = "Report generation failed."
    else:
        results['report'] = "Report generator not available."
        
    if patient_info:
        results['patient_info'] = patient_info
        
    # Generate visualization
    try:
        viz_image = create_enhanced_visualization(image, results)
        viz_base64 = f"data:image/png;base64,{image_to_base64(viz_image)}"
    except Exception as e:
        print(f"Error generating visualization: {e}")
        traceback.print_exc()
        viz_base64 = None
    
    # Remove non-serializable objects
    if 'masks' in results:
        del results['masks']
    if 'view_classification' in results and 'mask' in results['view_classification']:
        del results['view_classification']['mask']
        
    # Convert numpy types to native Python types
    safe_results = convert_numpy_types(results)

    response = {
        "results": safe_results,
        "visualization_base64": viz_base64,
        "original_base64": f"data:image/png;base64,{image_to_base64(image)}"
    }
    
    return response

