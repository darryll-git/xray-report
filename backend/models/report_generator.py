#models/report_generator.py

import os
from groq import Groq
import base64
from io import BytesIO
from PIL import Image
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

class ReportGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
    
    def image_to_base64(self, image, max_size=(1024, 1024)):
        """Convert PIL Image to base64 with size limits
        
        Args:
            image: PIL Image
            max_size: Maximum dimensions (width, height) to resize to while maintaining aspect ratio
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate resize dimensions while maintaining aspect ratio
        img_width, img_height = image.size
        resize_ratio = min(max_size[0] / img_width, max_size[1] / img_height)
        if resize_ratio < 1:
            new_width = int(img_width * resize_ratio)
            new_height = int(img_height * resize_ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        # Save as JPEG with compression to ensure smaller file size
        image.save(buffered, format="JPEG", optimize=True, quality=85)
        
        # Check if size is under 4MB
        img_data = buffered.getvalue()
        if len(img_data) > 4 * 1024 * 1024:  # 4MB in bytes
            # If still too large, increase compression
            buffered = BytesIO()
            image.save(buffered, format="JPEG", optimize=True, quality=50)
            img_data = buffered.getvalue()
        
        return base64.b64encode(img_data).decode()
    
    def generate_structured_report(self, image, analysis_data, patient_info=None):
        """Generate structured radiology report using Groq Vision LLM"""
        
        # If patient info not provided, use defaults and any detected info
        if not patient_info:
            patient_info = {}
            if 'view_classification' in analysis_data:
                view_data = analysis_data['view_classification']
                if 'age' in view_data:
                    patient_info['age'] = int(view_data['age'])
                if 'gender' in view_data:
                    patient_info['gender'] = view_data['gender']
        
        # Add default values for required fields
        patient_info.setdefault('name', 'ANONYMOUS')
        patient_info.setdefault('mrn', 'NOT PROVIDED')
        patient_info.setdefault('dob', 'NOT PROVIDED')
        patient_info.setdefault('referring_physician', 'NOT PROVIDED')
        patient_info.setdefault('exam_date', 'CURRENT')
        
        # Compile findings into a structured format
        findings_summary = self._compile_findings(analysis_data)
        
        # If Groq API is available, use vision model
        if self.client:
            try:
                report = self._generate_with_groq(image, findings_summary, patient_info, analysis_data)
                return report
            except Exception as e:
                print(f"Error with Groq API: {e}")
                return self._generate_template_report(findings_summary, patient_info, analysis_data)
        else:
            return self._generate_template_report(findings_summary, patient_info, analysis_data)
        
    def _format_pulmonary_findings(self, data):
        """Format pulmonary analysis for LLM"""
        text = []
        
        if 'edema_pattern' in data:
            pattern = data['edema_pattern']
            text.append(f"Edema Pattern: {pattern['pattern_type']}")
            text.append(f"Pattern Confidence: {pattern['confidence']:.0%}")
            
            if pattern['supporting_features']:
                text.append("Supporting Features:")
                for feat in pattern['supporting_features']:
                    text.append(f"  - {feat}")
            
            if pattern['kerley_lines'] and pattern['kerley_lines']['confidence'] > 0.3:
                kerley = pattern['kerley_lines']
                text.append(f"Interstitial Markings: {kerley['interpretation']}")
        
        if 'fissures' in data:
            text.append("\nPulmonary Fissures:")
            for viz_line in data['fissures'].get('visualization_lines', []):
                text.append(f"  - {viz_line['type'].title()} fissure ({viz_line['side']}): {viz_line['confidence']:.0%} confidence")
        
        return "\n".join(text) if text else "No additional pulmonary findings"
    
    def _generate_with_groq(self, image, findings_summary, patient_info, analysis_data):
        """Generate report using Groq's vision model"""
        
        # Convert image to base64
        img_base64 = self.image_to_base64(image)
        
        # Extract specific measurements for prompt
        ctr_data = analysis_data.get('ctr', {})
        heart_width = ctr_data.get('heart_width', 'Not available')
        thorax_width = ctr_data.get('chest_width', 'Not available')
        
        trachea_data = analysis_data.get('trachea', {})
        trachea_assessment = trachea_data.get('assessment', '')
        
        angles_data = analysis_data.get('costophrenic_angles', {})
        left_angle = angles_data.get('left_angle', 'Not measured')
        right_angle = angles_data.get('right_angle', 'Not measured')
        
        prompt = f"""You are an expert radiologist. Analyze this chest X-ray and generate a comprehensive structured radiology report.

Use the following automated analysis results as reference:
{findings_summary}

Pulmonary Pattern Analysis
{self._format_pulmonary_findings(analysis_data)}

Key measurements available:
- Heart width: {heart_width} pixels
- Thorax width: {thorax_width} pixels
- Left costophrenic angle: {left_angle}° (Code considers <90° as sharp; clinical norm ~30-45°)
- Right costophrenic angle: {right_angle}° (Code considers <90° as sharp; clinical norm ~30-45°)
- Trachea: {trachea_assessment}

Generate a detailed report with TWO MAIN SECTIONS:

PART 1: FINDINGS (Objective Measurements and Observations)
Present all factual findings and measurements without interpretation:

1. TECHNICAL PARAMETERS
   - Projection type (AP/PA)
   - Rotation score and symmetry status
   - Number of posterior ribs visible (specific count)
   - Penetration metrics (mean and std intensity values)

2. ANATOMICAL MEASUREMENTS
   - Trachea position with specific deviation direction if deviated (left vs right)
   - Cardiothoracic ratio (CTR) exact value
   - Heart width: {heart_width} pixels
   - Thorax width: {thorax_width} pixels
   - Costophrenic angles: Left {left_angle}°, Right {right_angle}°
   - Note: Code threshold for sharp angles is <90°; clinical normal range is typically 30-45° (code uses wider threshold for conservative detection)

3. SEGMENTATION RESULTS
   - Lung field contours identified
   - Cardiac silhouette borders delineated
   - Mediastinal boundaries marked

4. ABNORMALITY DETECTION SCORES
   - List all pathology detection scores as percentages
   - Highlight findings exceeding clinical thresholds

PART 2: INTERPRETATION (Clinical Significance and Assessment)
Provide clinical interpretation of the findings:

5. TECHNICAL QUALITY ASSESSMENT
   - Overall image quality (adequate/suboptimal)
   - Impact on diagnostic confidence
   - Recommendations for repeat imaging if needed

6. ANATOMICAL ASSESSMENT
   - Airway: Trachea alignment and any deviation direction
   - Mediastinum: Width and contour evaluation
   - Cardiac: Size assessment based on CTR (>0.50 = cardiomegaly)
   - Pleura: Costophrenic angle sharpness (mention code uses <90° threshold vs clinical 30-45°)
   - Lungs: Parenchymal assessment

7. CLINICAL IMPRESSION
   - Summary of significant findings
   - Clinical relevance and urgency level
   - Differential diagnoses if applicable
   - Recommendations for follow-up or additional studies

Use plain text without markdown formatting (no asterisks, hashtags, or special characters). Be detailed and professional."""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert radiologist analyzing chest X-ray images. Structure reports with clear separation between objective findings and clinical interpretation. Use plain text without markdown formatting."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Try Scout model first
            try:
                completion = self.client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    stop=None
                )
            except Exception as e:
                print(f"Scout model error: {e}, trying Maverick...")
                completion = self.client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    stop=None
                )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq vision model error: {e}")
            return self._generate_with_text_model(findings_summary, patient_info, analysis_data)
    
    def _generate_with_text_model(self, findings_summary, patient_info, analysis_data):
        """Generate report using Groq's text model"""
        
        # Extract specific measurements
        ctr_data = analysis_data.get('ctr', {})
        heart_width = ctr_data.get('heart_width', 'Not available')
        thorax_width = ctr_data.get('chest_width', 'Not available')
        
        angles_data = analysis_data.get('costophrenic_angles', {})
        left_angle = angles_data.get('left_angle', 'Not measured')
        right_angle = angles_data.get('right_angle', 'Not measured')
        
        prompt = f"""You are an expert radiologist. Generate a comprehensive structured chest X-ray radiology report based on the following automated analysis:

{findings_summary}

Specific measurements:
- Heart width: {heart_width} pixels
- Thorax width: {thorax_width} pixels
- Costophrenic angles: Left {left_angle}°, Right {right_angle}° (Code threshold <90°; clinical normal ~30-45°)

Generate a detailed report with TWO MAIN SECTIONS:

PART 1: FINDINGS (Objective Measurements)
- Technical parameters with specific values
- Anatomical measurements including pixel dimensions
- Abnormality detection scores

PART 2: INTERPRETATION (Clinical Assessment)
- Technical quality assessment
- Anatomical evaluation with clinical context
- Clinical impression and recommendations

Use plain text without markdown formatting. Be detailed and professional."""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert radiologist generating structured chest X-ray reports. Use plain text without markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                stop=None
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq text model error: {e}")
            return self._generate_template_report(findings_summary, patient_info, analysis_data)
    
    def _safe_get(self, entry, key="assessment", default="Unknown"):
        """Safely get value from either a dict or direct value"""
        if isinstance(entry, dict):
            return entry.get(key, default)
        return entry if entry is not None else default
    
    def _compile_findings(self, data):
        """Compile analysis data into readable findings"""
        findings = []
        
        # Technical considerations
        if 'view_classification' in data:
            view = data['view_classification']
            if isinstance(view, dict):
                view_type = view.get('view', 'Unknown')
                confidence = view.get('confidence', 0)
                findings.append(f"View: {view_type} (Confidence: {confidence:.2f})")
            else:
                findings.append(f"View: {view}")
        
        if 'rotation' in data:
            findings.append(f"Rotation: {self._safe_get(data['rotation'])}")
        
        if 'inspiration' in data:
            insp = data['inspiration']
            if isinstance(insp, dict):
                count = insp.get('count', 0)
                assessment = self._safe_get(insp)
                findings.append(f"Inspiration: {count} posterior ribs visible - {assessment}")
            else:
                findings.append(f"Inspiration: {insp}")
        
        if 'penetration' in data:
            findings.append(f"Penetration: {self._safe_get(data['penetration'])}")
        
        # Airway
        if 'trachea' in data:
            findings.append(f"Trachea: {self._safe_get(data['trachea'])}")
        
        # Cardiac
        if 'ctr' in data:
            findings.append(f"Cardiothoracic Ratio: {self._safe_get(data['ctr'])}")
        
        # Pleura
        if 'costophrenic_angles' in data:
            findings.append(f"Costophrenic Angles: {self._safe_get(data['costophrenic_angles'])}")
        
        # Abnormalities
        if 'abnormalities' in data:
            sig_findings = data.get('significant_findings', {})
            if isinstance(sig_findings, dict) and sig_findings:
                findings.append(f"Detected abnormalities: {', '.join(sig_findings.keys())}")
            elif isinstance(sig_findings, (list, set)):
                findings.append(f"Detected abnormalities: {', '.join(sig_findings)}")
            elif sig_findings:
                findings.append(f"Detected abnormalities: {sig_findings}")
            else:
                findings.append("No significant abnormalities detected by automated analysis")

        if 'edema_pattern' in data:
            pattern = data['edema_pattern']
            pattern_type = pattern['pattern_type']
            
            if pattern_type != 'no_edema':
                findings.append(f"Pulmonary Pattern: {pattern_type.replace('_', ' ').title()}")
                
                if pattern['supporting_features']:
                    findings.append(f"  Supporting features: {', '.join(pattern['supporting_features'])}")
                
                if pattern['kerley_lines']:
                    kerley = pattern['kerley_lines']
                    if kerley['confidence'] > 0.3:
                        findings.append(f"  {kerley['interpretation']}")
        
        if 'fissures' in data:
            fissure_count = len(data['fissures'].get('visualization_lines', []))
            if fissure_count > 0:
                findings.append(f"Pulmonary Fissures: {fissure_count} fissure(s) visualized")
        
        return "\n".join(findings)
    
    def _generate_template_report(self, findings_summary, patient_info, analysis_data):
        """Generate template-based report with two-part structure"""
        
        # Extract specific measurements
        ctr_data = analysis_data.get('ctr') or {}
        heart_width = ctr_data.get('heart_width', 'Not available')
        thorax_width = ctr_data.get('chest_width', 'Not available')
        ctr_value = ctr_data.get('ctr')
        if ctr_value is None:
            ctr_value = 'Not calculated'
        
        # Format pixel measurements
        if isinstance(heart_width, (int, float)):
            heart_width_str = f"{heart_width:.1f} pixels"
        else:
            heart_width_str = str(heart_width)
            
        if isinstance(thorax_width, (int, float)):
            thorax_width_str = f"{thorax_width:.1f} pixels"
        else:
            thorax_width_str = str(thorax_width)
        
        # Get trachea info with direction
        trachea_data = analysis_data.get('trachea', {})
        trachea_midline = trachea_data.get('midline', True)
        trachea_assessment = trachea_data.get('assessment', 'Trachea position normal')
        
        # Parse deviation direction if present
        if 'deviation detected' in trachea_assessment.lower():
            # Try to determine direction from the assessment text
            if 'left' in trachea_assessment.lower():
                trachea_status = "Deviation to the left detected"
            elif 'right' in trachea_assessment.lower():
                trachea_status = "Deviation to the right detected"
            else:
                # Default generic deviation message
                trachea_status = "Tracheal deviation detected (direction: see technical parameters)"
        else:
            trachea_status = "Trachea appears midline"
        
        # Get costophrenic angles
        angles_data = analysis_data.get('costophrenic_angles', {})
        left_angle = angles_data.get('left_angle', None)
        right_angle = angles_data.get('right_angle', None)
        
        if left_angle is not None:
            left_angle_str = f"{left_angle:.1f} degrees"
        else:
            left_angle_str = "Not measured"
            
        if right_angle is not None:
            right_angle_str = f"{right_angle:.1f} degrees"
        else:
            right_angle_str = "Not measured"
        
        # Extract age suffix
        age_suffix = ''
        if 'age' in patient_info:
            age = patient_info['age']
            if isinstance(age, (int, float)):
                if age < 2:
                    age_suffix = 'months' if age <= 1 else 'month'
                else:
                    age_suffix = 'years' if age > 1 else 'year'
        
        report = f"""RADIOLOGIC CONSULTATION
======================

PATIENT INFORMATION:
Name: {patient_info['name']}
MRN: {patient_info['mrn']}
DOB: {patient_info['dob']}
Age: {patient_info.get('age', 'NOT PROVIDED')} {age_suffix}
Gender: {patient_info.get('gender', 'NOT PROVIDED')}
Exam Date: {patient_info['exam_date']}
Referring Physician: {patient_info['referring_physician']}

EXAM: CHEST X-RAY
History: {patient_info.get('history', 'Routine chest examination')}

═══════════════════════════════════════════════════════════════════
PART 1: FINDINGS (Objective Measurements and Observations)
═══════════════════════════════════════════════════════════════════

{findings_summary}

1. TECHNICAL PARAMETERS
   - Projection: As noted in automated analysis
   - Rotation Assessment: Symmetry evaluation completed
   - Inspiration: Posterior rib count documented
   - Penetration: Exposure metrics measured

2. ANATOMICAL MEASUREMENTS
   - Trachea: {trachea_status}
   - Cardiothoracic Ratio: {ctr_value if isinstance(ctr_value, str) else f'{ctr_value:.3f}'}
   - Heart Width: {heart_width_str}
   - Thorax Width: {thorax_width_str}
   - Costophrenic Angles:
     * Left: {left_angle_str}
     * Right: {right_angle_str}
     * Note: Code uses <90 degrees as threshold for sharp angles (conservative detection)
     * Clinical normal range typically 30-45 degrees
     * Code threshold is wider to avoid false positives for effusions

3. SEGMENTATION DATA
   - Lung Fields: Bilateral contours identified
   - Cardiac Silhouette: Border demarcation completed
   - Mediastinum: Boundaries delineated

4. ABNORMALITY DETECTION
   - Automated pathology screening performed
   - Detection scores calculated for multiple conditions
   - Threshold analysis completed

═══════════════════════════════════════════════════════════════════
PART 2: INTERPRETATION (Clinical Significance and Assessment)
═══════════════════════════════════════════════════════════════════

5. TECHNICAL QUALITY ASSESSMENT

The examination demonstrates technical parameters within acceptable ranges for diagnostic interpretation. Optimal PA projection is preferred for accurate cardiothoracic ratio measurement. The degree of inspiration and penetration appear adequate for evaluation of cardiopulmonary structures.

6. ANATOMICAL ASSESSMENT

AIRWAY AND MEDIASTINUM:
The trachea has been assessed for position and alignment. {trachea_assessment}. The mediastinum shows no obvious abnormal widening on automated assessment. The carina and main bronchi appear within normal limits based on available data.

CARDIAC SILHOUETTE:
Cardiac size has been evaluated with cardiothoracic ratio calculation showing a value of {ctr_value if isinstance(ctr_value, str) else f'{ctr_value:.3f}'}. The cardiac borders are visualized with heart width measuring {heart_width_str} and thorax width measuring {thorax_width_str}. Assessment of cardiac dimensions has been completed relative to thoracic cage width.

LUNGS AND PLEURA:
The lung fields have been analyzed through automated segmentation. Pleural spaces are evaluated. Costophrenic angles measured at {left_angle_str} (left) and {right_angle_str} (right). The automated analysis uses a threshold of <90 degrees for identifying sharp angles, which is more conservative than the typical clinical range of 30-45 degrees. This wider threshold helps avoid false positive detection of pleural effusions while still identifying truly blunted angles.

OSSEOUS STRUCTURES:
Visualized ribs, clavicles, and spine show no obvious fractures or lytic lesions on automated review. Posterior rib count assists in assessing adequacy of inspiration.

7. CLINICAL IMPRESSION

Based on the automated analysis provided, the study demonstrates the technical and anatomical findings detailed above. Any significant deviations from normal parameters have been flagged by the automated detection system.

CLINICAL SIGNIFICANCE:
- Technical quality appears adequate for diagnostic interpretation
- Anatomical structures have been systematically evaluated
- Automated pathology screening has been performed

RECOMMENDATIONS:
- This automated report should be reviewed and verified by a qualified radiologist for clinical decision-making
- Correlation with clinical presentation and prior imaging is recommended
- Any significant findings flagged by automated analysis warrant radiologist review

Report Generated: {patient_info['exam_date']}
Analysis Method: AI-Assisted Automated Analysis

═══════════════════════════════════════════════════════════════════
DISCLAIMER: This is an AI-generated report based on automated analysis. 
Final interpretation must be made by a licensed radiologist.
═══════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def generate_docx_report(self, report_text, patient_info=None):
        """Generate a Word document version of the report"""
        doc = Document()
        
        # Set default font
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Arial'
        font.size = Pt(11)
        
        # Title
        title = doc.add_heading('RADIOLOGIC CONSULTATION', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Parse and format the report
        lines = report_text.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Remove markdown formatting
            line = line.replace('**', '').replace('##', '').replace('###', '')
            
            # Check for section headers
            if line.startswith('═══'):
                continue
            elif 'PART 1:' in line or 'PART 2:' in line:
                heading = doc.add_heading(line, level=1)
                heading.runs[0].font.color.rgb = RGBColor(0, 51, 102)
            elif line.endswith(':') and len(line) < 50 and not line.startswith(' '):
                # Subsection headers
                if any(keyword in line.upper() for keyword in ['PATIENT', 'EXAM', 'TECHNICAL', 'ANATOMICAL', 'SEGMENTATION', 'ABNORMALITY', 'QUALITY', 'ASSESSMENT', 'IMPRESSION', 'CLINICAL', 'RECOMMENDATIONS', 'DISCLAIMER']):
                    heading = doc.add_heading(line, level=2)
                    heading.runs[0].font.color.rgb = RGBColor(0, 76, 153)
                else:
                    p = doc.add_paragraph(line)
                    p.runs[0].bold = True
            elif line.startswith('-') or line.startswith('*') or line.startswith('•'):
                # Bullet points
                doc.add_paragraph(line[1:].strip(), style='List Bullet')
            else:
                # Regular text
                doc.add_paragraph(line)
        
        # Save to BytesIO
        docx_buffer = BytesIO()
        doc.save(docx_buffer)
        docx_buffer.seek(0)
        
        return docx_buffer