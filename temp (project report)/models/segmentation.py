#models/segmentation.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import numpy as np
from PIL import Image
import cv2

class AnatomySegmenter:
    def __init__(self, device='cpu', view_classifier=None):
        self.device = device
        self.lung_model = None
        self.heart_model = None
        self.view_classifier = view_classifier  # Can use ianpan model for segmentation
        
    def segment_all(self, image, use_ml_model=True, ml_masks=None):
        """Segment all anatomical structures
        
        Args:
            image: Input image
            use_ml_model: If True and ml_masks available, use ML segmentation
            ml_masks: Pre-computed masks from ianpan model (if available)
        """
        if use_ml_model and ml_masks is not None:
            # Convert image to proper size for mask creation
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                img_array = image
                
            # Ensure ml_masks has same dimensions as image
            ml_masks = cv2.resize(ml_masks, (img_array.shape[1], img_array.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Create lung mask (combine right and left lungs)
            lung_mask = np.zeros_like(img_array, dtype=np.uint8)
            lung_mask[ml_masks == 1] = 255  # Right lung
            lung_mask[ml_masks == 2] = 255  # Left lung
            
            # Create heart mask
            heart_mask = np.zeros_like(img_array, dtype=np.uint8)
            heart_mask[ml_masks == 3] = 255  # Heart
            
            mediastinum_mask = self.segment_mediastinum(image, lung_mask)
            
            return {
                'lungs': lung_mask.astype(np.uint8),
                'heart': heart_mask.astype(np.uint8),
                'mediastinum': mediastinum_mask,
                'method': 'ml_model'
            }
        else:
            # Use traditional segmentation
            lung_mask = self.segment_lungs_basic(image)
            heart_mask = self.segment_heart_basic(image, lung_mask)
            mediastinum_mask = self.segment_mediastinum(image, lung_mask)
            
            return {
                'lungs': lung_mask,
                'heart': heart_mask,
                'mediastinum': mediastinum_mask,
                'method': 'traditional'
            }
        
    def load_lung_segmentation(self):
        """Load lung segmentation model"""
        try:
            from huggingface_hub import hf_hub_download
            # Using a simple UNet approach or pretrained model
            # For production, use maja011235/lung-segmentation-unet
            print("Loading lung segmentation model...")
            self.lung_model = "loaded"  # Placeholder for actual model
        except Exception as e:
            print(f"Error loading lung model: {e}")
            self.lung_model = None
    
    def segment_lungs_basic(self, image):
        """Basic lung segmentation using thresholding and morphology"""
        # Convert to grayscale
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        else:
            img_gray = image
        
        # Normalize
        img_norm = ((img_gray - img_gray.min()) * (255.0 / (img_gray.max() - img_gray.min()))).astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_norm, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert (lungs should be white)
        binary = cv2.bitwise_not(binary)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours and keep largest two (left and right lung)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        lung_mask = np.zeros_like(img_gray)
        if len(contours) >= 2:
            # Sort by area and take top 2
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            cv2.drawContours(lung_mask, contours, -1, (255,), -1)
        
        return lung_mask
    
    def segment_heart_basic(self, image, lung_mask):
        """Basic heart segmentation"""
        # Convert image to grayscale numpy array
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        else:
            img_gray = image.copy()  # Make a copy to avoid modifying original
            
        # Ensure lung_mask has same dimensions and type
        lung_mask = cv2.resize(lung_mask, (img_gray.shape[1], img_gray.shape[0]))
        lung_mask = lung_mask.astype(np.uint8)
        
        h, w = img_gray.shape
        
        # Focus on central-lower region where heart typically is
        roi_y1, roi_y2 = int(h * 0.4), int(h * 0.8)
        roi_x1, roi_x2 = int(w * 0.3), int(w * 0.7)
        
        roi = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Thresholding to find dense areas
        _, binary = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Create full-size mask
        heart_mask = np.zeros_like(img_gray, dtype=np.uint8)
        heart_mask[roi_y1:roi_y2, roi_x1:roi_x2] = closing
        
        # Remove overlap with lungs - ensure masks are binary
        inv_lung_mask = cv2.bitwise_not(lung_mask)
        heart_mask = cv2.bitwise_and(heart_mask, inv_lung_mask)
        
        return heart_mask
    
    def segment_mediastinum(self, image, lung_mask):
        """Segment mediastinum (space between lungs)"""
        # Ensure lung_mask and image have same dimensions
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        else:
            img_gray = image.copy()  # Make a copy to avoid modifying original
            
        # Ensure lung_mask has same dimensions and type as image
        lung_mask = cv2.resize(lung_mask, (img_gray.shape[1], img_gray.shape[0]))
        lung_mask = lung_mask.astype(np.uint8)
        
        h, w = img_gray.shape
        
        # Create a mask for the central region
        mediastinum_mask = np.zeros_like(img_gray, dtype=np.uint8)
        center_x = w // 2
        margin = int(w * 0.1)  # 10% of width on each side of center
        
        mediastinum_mask[:, center_x - margin:center_x + margin] = 255
        
        # Remove lung areas - ensure both masks are binary and same size
        inv_lung_mask = cv2.bitwise_not(lung_mask)
        mediastinum_mask = cv2.bitwise_and(mediastinum_mask, inv_lung_mask)
        
        return mediastinum_mask
    
    def calculate_ctr_from_mask(self, mask):
        """
        Balanced CTR calculation using robust contours:
        - Uses outer lung borders but filters small noisy regions.
        - Uses convex hull for heart.
        - Matches clinical CTR range (~0.45–0.60 for most normal X-rays).
        """

        lungs = np.zeros_like(mask)
        lungs[(mask == 1) | (mask == 2)] = 255
        heart = np.zeros_like(mask)
        heart[mask == 3] = 255

        # --- 1. LUNG CONTOURS ---
        lung_contours, _ = cv2.findContours(lungs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(lung_contours) < 2:
            return {'ctr': None, 'interpretation': 'Failed: insufficient lung contours'}

        # Sort by area and keep top 2
        lung_contours = sorted(lung_contours, key=cv2.contourArea, reverse=True)[:2]
        lung_contours = [cv2.convexHull(c) for c in lung_contours]

        # Get leftmost and rightmost x from both lungs (outer borders)
        lung_left_edge = min([c[:, 0, 0].min() for c in lung_contours])
        lung_right_edge = max([c[:, 0, 0].max() for c in lung_contours])
        chest_width = lung_right_edge - lung_left_edge

        if chest_width <= 0:
            return {'ctr': None, 'interpretation': 'Failed: invalid chest width'}

        midline = (lung_left_edge + lung_right_edge) / 2.0

        # --- 2. HEART CONTOUR ---
        heart_contours, _ = cv2.findContours(heart.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(heart_contours) == 0:
            return {'ctr': None, 'interpretation': 'Failed: no heart contour'}

        heart_c = max(heart_contours, key=cv2.contourArea)
        heart_c = cv2.convexHull(heart_c)

        heart_left = heart_c[:, 0, 0].min()
        heart_right = heart_c[:, 0, 0].max()

        # --- 3. Heart projections from midline ---
        left_proj = abs(midline - heart_left)
        right_proj = abs(heart_right - midline)
        heart_width = left_proj + right_proj

        # --- 4. CTR ---
        ctr = heart_width / chest_width

        # --- 5. Interpretation ---
        if ctr > 0.55:
            interp = f"Marked cardiomegaly (CTR={ctr:.2f})"
        elif ctr > 0.50:
            interp = f"Mild cardiomegaly (CTR={ctr:.2f})"
        else:
            interp = f"Normal (CTR={ctr:.2f})"

        return {
            'ctr': float(ctr),
            'interpretation': interp,
            'heart_width': float(heart_width),
            'chest_width': float(chest_width),
            'midline': float(midline),
            'heart_left_edge': int(heart_left),
            'heart_right_edge': int(heart_right),
            'lung_left_edge': int(lung_left_edge),
            'lung_right_edge': int(lung_right_edge)
        }

    
    def get_lung_contours(self, lung_mask):
        """Extract lung contours"""
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[:2] if len(contours) >= 2 else contours