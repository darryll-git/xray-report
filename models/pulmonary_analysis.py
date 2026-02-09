#models/pulmonary_analysis.py

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

class PulmonaryEdemaAnalyzer:
    """
    Analyzes pulmonary edema patterns without relying on CTR.
    Focuses on distribution patterns and interstitial markings.
    """
    
    def __init__(self):
        self.kerley_confidence_threshold = 0.3  # Conservative threshold
        
    def analyze_edema_pattern(self, image, lung_mask, pathology_scores):
        """
        Analyze cardiogenic vs non-cardiogenic edema patterns
        
        Args:
            image: Grayscale chest X-ray image
            lung_mask: Binary mask of lung fields
            pathology_scores: Dictionary from TorchXRayVision
            
        Returns:
            Dictionary with edema pattern analysis
        """
        result = {
            'pattern_type': None,
            'confidence': 0.0,
            'supporting_features': [],
            'kerley_lines': None,
            'distribution': None
        }
        
        # Check if edema is even detected
        edema_score = pathology_scores.get('Edema', 0.0)
        if edema_score < 0.5:
            result['pattern_type'] = 'no_edema'
            result['confidence'] = 1.0 - edema_score
            return result
        
        # Analyze distribution pattern
        distribution = self._analyze_distribution(image, lung_mask)
        result['distribution'] = distribution
        
        # Detect Kerley-like lines
        kerley_analysis = self._detect_kerley_lines(image, lung_mask)
        result['kerley_lines'] = kerley_analysis
        
        # Get supporting pathology indicators
        supporting_features = []
        
        # Check for bilateral pattern
        if distribution['bilateral'] and distribution['symmetry_score'] > 0.7:
            supporting_features.append('bilateral symmetric opacities')
        
        # Check for perihilar pattern
        if distribution['perihilar_dominant']:
            supporting_features.append('perihilar (bat-wing) pattern')
        
        # Check for interstitial markings
        if kerley_analysis['b_lines_detected'] or kerley_analysis['a_lines_detected']:
            if kerley_analysis['confidence'] > self.kerley_confidence_threshold:
                supporting_features.append('interstitial linear markings suggestive of Kerley-type lines')
        
        # Check for pleural effusion
        effusion_score = pathology_scores.get('Effusion', 0.0)
        if effusion_score > 0.6:
            supporting_features.append('bilateral pleural effusions')
        
        # Check for cardiomegaly probability (not CTR measurement)
        cardiomegaly_score = pathology_scores.get('Cardiomegaly', 0.0)
        if cardiomegaly_score > 0.6:
            supporting_features.append('cardiac silhouette enlargement')
        
        # Determine pattern type based on features
        cardiogenic_indicators = sum([
            distribution['bilateral'] and distribution['symmetry_score'] > 0.7,
            distribution['perihilar_dominant'],
            effusion_score > 0.6,
            cardiomegaly_score > 0.6
        ])
        
        if cardiogenic_indicators >= 2:
            result['pattern_type'] = 'cardiogenic'
            result['confidence'] = min(0.9, 0.5 + (cardiogenic_indicators * 0.15))
        elif cardiogenic_indicators == 1:
            result['pattern_type'] = 'indeterminate'
            result['confidence'] = 0.5
        else:
            result['pattern_type'] = 'non_cardiogenic_pattern'
            result['confidence'] = 0.6
        
        result['supporting_features'] = supporting_features
        
        return result
    
    def _analyze_distribution(self, image, lung_mask):
        """Analyze opacity distribution patterns"""
        h, w = image.shape
        
        # Separate left and right lungs
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return {
                'bilateral': False,
                'symmetry_score': 0.0,
                'perihilar_dominant': False,
                'peripheral_dominant': False
            }
        
        # Sort contours by x position
        contours = sorted(contours, key=lambda c: c[:, 0, 0].mean())[:2]
        
        # Create separate masks
        left_lung = np.zeros_like(lung_mask)
        right_lung = np.zeros_like(lung_mask)
        cv2.drawContours(left_lung, [contours[0]], -1, 255, -1)
        cv2.drawContours(right_lung, [contours[1]], -1, 255, -1)
        
        # Calculate mean intensities (lower = more opacity)
        left_intensity = np.mean(image[left_lung > 0])
        right_intensity = np.mean(image[right_lung > 0])
        
        # Symmetry score (closer to 1 = more symmetric)
        symmetry_score = 1.0 - abs(left_intensity - right_intensity) / max(left_intensity, right_intensity)
        
        # Detect bilateral involvement
        bilateral = (left_intensity < 120 and right_intensity < 120)
        
        # Analyze central vs peripheral distribution
        center_x = w // 2
        perihilar_region = image[:, int(center_x - w*0.2):int(center_x + w*0.2)]
        perihilar_mask = lung_mask[:, int(center_x - w*0.2):int(center_x + w*0.2)]
        
        if perihilar_mask.sum() > 0:
            perihilar_intensity = np.mean(perihilar_region[perihilar_mask > 0])
            overall_intensity = np.mean(image[lung_mask > 0])
            
            # Perihilar dominant if central regions are darker
            perihilar_dominant = perihilar_intensity < overall_intensity * 0.9
        else:
            perihilar_dominant = False
        
        return {
            'bilateral': bilateral,
            'symmetry_score': float(symmetry_score),
            'perihilar_dominant': perihilar_dominant,
            'peripheral_dominant': not perihilar_dominant,
            'left_intensity': float(left_intensity),
            'right_intensity': float(right_intensity)
        }
    
    def _detect_kerley_lines(self, image, lung_mask):
        """
        Detect Kerley A and B line patterns
        These are probabilistic detections, not definitive
        """
        # Enhance edges
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Apply lung mask
        edges_masked = cv2.bitwise_and(edges, edges, mask=lung_mask)
        
        # Detect Kerley B lines (short horizontal lines at periphery)
        b_lines = self._detect_b_lines(edges_masked, lung_mask)
        
        # Detect Kerley A lines (longer oblique lines)
        a_lines = self._detect_a_lines(edges_masked, lung_mask)
        
        # Calculate overall confidence
        confidence = (b_lines['count'] * 0.05 + a_lines['count'] * 0.08)
        confidence = min(confidence, 0.8)  # Cap at 0.8, never certain
        
        return {
            'b_lines_detected': b_lines['count'] > 0,
            'b_line_count': b_lines['count'],
            'b_line_regions': b_lines['regions'],
            'a_lines_detected': a_lines['count'] > 0,
            'a_line_count': a_lines['count'],
            'a_line_regions': a_lines['regions'],
            'confidence': float(confidence),
            'interpretation': self._interpret_kerley_findings(b_lines['count'], a_lines['count'], confidence)
        }
    
    def _detect_b_lines(self, edges, lung_mask):
        """Detect Kerley B line patterns (peripheral short horizontal lines)"""
        h, w = edges.shape
        
        # Focus on peripheral lung bases
        peripheral_region = edges.copy()
        center_mask = np.zeros_like(lung_mask)
        
        # Create peripheral mask (exclude central 40%)
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Erode to get peripheral zone
            eroded = cv2.erode(lung_mask, np.ones((50, 50), np.uint8), iterations=1)
            center_mask = cv2.bitwise_or(center_mask, eroded)
        
        peripheral_mask = cv2.bitwise_xor(lung_mask, center_mask)
        peripheral_region = cv2.bitwise_and(peripheral_region, peripheral_region, mask=peripheral_mask)
        
        # Focus on lower 60% (lung bases)
        peripheral_region[:int(h*0.4), :] = 0
        
        # Detect horizontal lines using HoughLinesP
        lines = cv2.HoughLinesP(peripheral_region, 1, np.pi/180, threshold=20, 
                                minLineLength=10, maxLineGap=5)
        
        b_line_regions = []
        b_line_count = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                
                # Kerley B: short (~10-20mm), horizontal (0-20 degrees)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if 10 < length < 40 and (angle < 20 or angle > 160):
                    b_line_count += 1
                    b_line_regions.append({
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'length': float(length),
                        'angle': float(angle)
                    })
        
        return {
            'count': b_line_count,
            'regions': b_line_regions
        }
    
    def _detect_a_lines(self, edges, lung_mask):
        """Detect Kerley A line patterns (longer oblique lines)"""
        h, w = edges.shape
        
        # Kerley A lines: 2-6cm, oblique, extend toward hilum
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=50, maxLineGap=10)
        
        a_line_regions = []
        a_line_count = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                
                # Kerley A: longer, oblique (20-70 degrees from horizontal)
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if 50 < length < 120 and 20 < angle < 70:
                    a_line_count += 1
                    a_line_regions.append({
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'length': float(length),
                        'angle': float(angle)
                    })
        
        return {
            'count': a_line_count,
            'regions': a_line_regions
        }
    
    def _interpret_kerley_findings(self, b_count, a_count, confidence):
        """Generate clinical interpretation of Kerley line findings"""
        if b_count == 0 and a_count == 0:
            return "No interstitial linear markings detected"
        
        findings = []
        if b_count > 0:
            findings.append(f"{b_count} short peripheral linear opacities")
        if a_count > 0:
            findings.append(f"{a_count} longer oblique linear opacities")
        
        interpretation = f"Linear interstitial markings detected ({', '.join(findings)}), "
        
        if confidence > 0.5:
            interpretation += "which may represent Kerley-type lines in the appropriate clinical context"
        else:
            interpretation += "suggestive of interstitial thickening"
        
        return interpretation


class LungFissureDetector:
    """
    Detects major and minor pulmonary fissures
    This is pure anatomy - safe for visualization
    """
    
    def __init__(self):
        pass
    
    def detect_fissures(self, image, lung_mask):
        """
        Detect major (oblique) and minor (horizontal) fissures
        
        Args:
            image: Grayscale chest X-ray
            lung_mask: Binary lung mask
            
        Returns:
            Dictionary with fissure locations and confidence
        """
        result = {
            'major_fissures': [],
            'minor_fissure': None,
            'visualization_lines': []
        }
        
        # Enhance image for fissure detection
        enhanced = self._enhance_for_fissures(image, lung_mask)
        
        # Detect fissures using edge detection and line finding
        major_fissures = self._detect_major_fissures(enhanced, lung_mask)
        minor_fissure = self._detect_minor_fissure(enhanced, lung_mask)
        
        result['major_fissures'] = major_fissures
        result['minor_fissure'] = minor_fissure
        
        # Create visualization lines
        viz_lines = []
        
        # Major fissures (both lungs)
        for fissure in major_fissures:
            if fissure['confidence'] > 0.3:
                viz_lines.append({
                    'type': 'major',
                    'side': fissure['side'],
                    'line': fissure['line'],
                    'confidence': fissure['confidence']
                })
        
        # Minor fissure (right lung only)
        if minor_fissure and minor_fissure['confidence'] > 0.3:
            viz_lines.append({
                'type': 'minor',
                'side': 'right',
                'line': minor_fissure['line'],
                'confidence': minor_fissure['confidence']
            })
        
        result['visualization_lines'] = viz_lines
        
        return result
    
    def _enhance_for_fissures(self, image, lung_mask):
        """Enhance image to make fissures more visible"""
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Apply lung mask
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=lung_mask)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 20, 60)
        
        return edges
    
    def _detect_major_fissures(self, edges, lung_mask):
        """
        Detect major (oblique) fissures
        Expected: oblique lines in both lungs
        """
        h, w = edges.shape
        
        # Separate lungs
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return []
        
        contours = sorted(contours, key=lambda c: c[:, 0, 0].mean())[:2]
        
        major_fissures = []
        
        for idx, contour in enumerate(contours):
            side = 'left' if idx == 0 else 'right'
            
            # Create mask for this lung
            lung_side_mask = np.zeros_like(lung_mask)
            cv2.drawContours(lung_side_mask, [contour], -1, 255, -1)
            
            # Apply mask to edges
            lung_edges = cv2.bitwise_and(edges, edges, mask=lung_side_mask)
            
            # Detect oblique lines (30-60 degrees)
            lines = cv2.HoughLinesP(lung_edges, 1, np.pi/180, threshold=30,
                                   minLineLength=int(h*0.2), maxLineGap=20)
            
            if lines is not None:
                # Find best oblique line
                best_line = None
                best_score = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # Major fissure: oblique (30-60 degrees), long
                    if 30 < angle < 60 and length > h*0.2:
                        score = length * (1 - abs(45 - angle) / 45)  # Prefer ~45 degrees
                        if score > best_score:
                            best_score = score
                            best_line = line[0]
                
                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    confidence = min(best_score / (h * 0.5), 0.9)
                    
                    major_fissures.append({
                        'side': side,
                        'line': {
                            'start': (int(x1), int(y1)),
                            'end': (int(x2), int(y2))
                        },
                        'confidence': float(confidence)
                    })
        
        return major_fissures
    
    def _detect_minor_fissure(self, edges, lung_mask):
        """
        Detect minor (horizontal) fissure
        Expected: horizontal line in right lung only
        """
        h, w = edges.shape
        
        # Get right lung (rightmost contour)
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return None
        
        contours = sorted(contours, key=lambda c: c[:, 0, 0].mean())
        right_lung_contour = contours[-1]  # Rightmost
        
        # Create right lung mask
        right_lung_mask = np.zeros_like(lung_mask)
        cv2.drawContours(right_lung_mask, [right_lung_contour], -1, 255, -1)
        
        # Focus on mid-upper region (where horizontal fissure typically is)
        roi_y1, roi_y2 = int(h * 0.3), int(h * 0.6)
        right_lung_mask[:roi_y1, :] = 0
        right_lung_mask[roi_y2:, :] = 0
        
        # Apply mask
        lung_edges = cv2.bitwise_and(edges, edges, mask=right_lung_mask)
        
        # Detect horizontal lines
        lines = cv2.HoughLinesP(lung_edges, 1, np.pi/180, threshold=25,
                               minLineLength=int(w*0.15), maxLineGap=15)
        
        if lines is not None:
            best_line = None
            best_score = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Minor fissure: nearly horizontal (0-15 degrees)
                if angle < 15 or angle > 165:
                    score = length * (1 - angle / 15)  # Prefer more horizontal
                    if score > best_score:
                        best_score = score
                        best_line = line[0]
            
            if best_line is not None:
                x1, y1, x2, y2 = best_line
                confidence = min(best_score / (w * 0.3), 0.8)
                
                return {
                    'line': {
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2))
                    },
                    'confidence': float(confidence)
                }
        
        return None