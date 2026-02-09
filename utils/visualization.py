#utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4):
    """Overlay segmentation mask on image"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    mask_colored = np.zeros_like(img_array)
    mask_colored[mask > 0] = color
    
    overlayed = cv2.addWeighted(img_array, 1, mask_colored, alpha, 0)
    return Image.fromarray(overlayed)

def draw_measurements(image, measurements):
    """Draw measurement lines and annotations"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Draw cardiothoracic ratio lines if available
    if 'heart_width' in measurements and 'thorax_width' in measurements:
        h, w = img_array.shape[:2]
        heart_w = measurements['heart_width']
        thorax_w = measurements['thorax_width']
        
        # Draw horizontal lines
        y_center = h // 2
        cv2.line(img_array, (w//2 - heart_w//2, y_center), 
                (w//2 + heart_w//2, y_center), (0, 255, 0), 2)
        cv2.line(img_array, (w//2 - thorax_w//2, y_center + 50), 
                (w//2 + thorax_w//2, y_center + 50), (255, 0, 0), 2)
    
    return Image.fromarray(img_array)

def draw_rib_markers(image, rib_positions):
    """Mark detected ribs"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    for i, pos in enumerate(rib_positions):
        cv2.circle(img_array, (int(pos[0]), int(pos[1])), 5, (0, 255, 255), -1)
        cv2.putText(img_array, f'R{i+1}', (int(pos[0])+10, int(pos[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return Image.fromarray(img_array)

def draw_fissures(image, fissure_data, alpha=0.7):
    """
    Draw pulmonary fissures with anatomical labels
    
    Args:
        image: PIL Image or numpy array
        fissure_data: Dictionary from LungFissureDetector
        alpha: Transparency for annotations
    """
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    overlay = img_array.copy()
    
    for viz_line in fissure_data.get('visualization_lines', []):
        line_type = viz_line['type']
        side = viz_line['side']
        line_coords = viz_line['line']
        confidence = viz_line['confidence']
        
        # Choose color based on type
        if line_type == 'major':
            color = (100, 200, 255)  # Light blue
            label = f"Major fissure ({side})"
        else:
            color = (150, 100, 255)  # Purple
            label = f"Minor fissure"
        
        # Draw dashed line
        start = line_coords['start']
        end = line_coords['end']
        
        # Draw dashed line effect
        draw_dashed_line(overlay, start, end, color, thickness=2, dash_length=10)
        
        # Add label
        mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        
        # Add background rectangle for text
        text = f"{label} ({confidence:.0%})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cv2.rectangle(overlay,
                     (mid_point[0] - 5, mid_point[1] - text_size[1] - 5),
                     (mid_point[0] + text_size[0] + 5, mid_point[1] + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(overlay, text, mid_point,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Blend overlay with original
    result = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
    
    return Image.fromarray(result)

def draw_kerley_regions(image, kerley_data, alpha=0.5):
    """
    Draw regions where Kerley lines are detected
    Uses subtle overlays, not exact line reproduction
    
    Args:
        image: PIL Image or numpy array
        kerley_data: Dictionary from PulmonaryEdemaAnalyzer
        alpha: Transparency for highlighting
    """
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    overlay = img_array.copy()
    
    # Only visualize if confidence is reasonable
    if kerley_data['confidence'] < 0.3:
        return Image.fromarray(img_array)
    
    # Draw B-line regions (peripheral bases)
    for b_region in kerley_data.get('b_line_regions', [])[:5]:  # Limit to 5 most prominent
        start = b_region['start']
        end = b_region['end']
        
        # Draw very faint line
        cv2.line(overlay, start, end, (255, 200, 100), 2, cv2.LINE_AA)
        
        # Draw small circle at midpoint
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.circle(overlay, mid, 3, (255, 200, 100), -1)
    
    # Draw A-line regions
    for a_region in kerley_data.get('a_line_regions', [])[:3]:  # Limit to 3
        start = a_region['start']
        end = a_region['end']
        
        # Draw faint line
        cv2.line(overlay, start, end, (100, 255, 200), 2, cv2.LINE_AA)
    
    # Add legend in corner
    h, w = img_array.shape[:2]
    
    if kerley_data.get('b_lines_detected') or kerley_data.get('a_lines_detected'):
        legend_y = h - 80
        cv2.rectangle(overlay, (10, legend_y), (300, h - 10), (0, 0, 0), -1)
        
        text_y = legend_y + 20
        cv2.putText(overlay, "Interstitial marking regions:",
                   (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if kerley_data.get('b_lines_detected'):
            cv2.circle(overlay, (30, text_y + 20), 3, (255, 200, 100), -1)
            cv2.putText(overlay, f"Short peripheral lines (n={kerley_data['b_line_count']})",
                       (40, text_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        if kerley_data.get('a_lines_detected'):
            cv2.circle(overlay, (30, text_y + 40), 3, (100, 255, 200), -1)
            cv2.putText(overlay, f"Longer oblique lines (n={kerley_data['a_line_count']})",
                       (40, text_y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1)
    
    # Blend
    result = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
    
    return Image.fromarray(result)

def draw_dashed_line(img, start, end, color, thickness=2, dash_length=10):
    """Draw a dashed line on the image"""
    dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    dashes = int(dist / dash_length)
    
    for i in range(dashes):
        if i % 2 == 0:  # Draw dash
            s = (
                int(start[0] + (end[0] - start[0]) * i / dashes),
                int(start[1] + (end[1] - start[1]) * i / dashes)
            )
            e = (
                int(start[0] + (end[0] - start[0]) * (i + 0.5) / dashes),
                int(start[1] + (end[1] - start[1]) * (i + 0.5) / dashes)
            )
            cv2.line(img, s, e, color, thickness, cv2.LINE_AA)

def create_enhanced_visualization(image, results):
    """
    Create comprehensive visualization with all analysis overlays
    
    Args:
        image: Original PIL Image
        results: Complete analysis results dictionary
    """
    img_array = np.array(image.convert('RGB'))
    masks = results['masks']
    
    # Start with base segmentation
    segmented_img = img_array.copy()
    
    # Overlay lungs in blue (transparent)
    lung_colored = np.zeros_like(img_array)
    lung_colored[masks['lungs'] > 0] = [0, 0, 255]
    segmented_img = np.where(lung_colored > 0, 
                             (segmented_img * 0.7 + lung_colored * 0.3).astype(np.uint8),
                             segmented_img)
    
    # Overlay heart in red (transparent)
    heart_colored = np.zeros_like(img_array)
    heart_colored[masks['heart'] > 0] = [255, 0, 0]
    segmented_img = np.where(heart_colored > 0,
                            (segmented_img * 0.7 + heart_colored * 0.3).astype(np.uint8),
                            segmented_img)
    
    # Convert to PIL for overlay functions
    working_image = Image.fromarray(segmented_img)
    
    # Draw fissures if detected
    if 'fissures' in results:
        working_image = draw_fissures(working_image, results['fissures'], alpha=0.6)
    
    # Draw Kerley line regions if detected (very subtle)
    if 'edema_pattern' in results and results['edema_pattern'].get('kerley_lines'):
        kerley_data = results['edema_pattern']['kerley_lines']
        if kerley_data['confidence'] > 0.3:
            working_image = draw_kerley_regions(working_image, kerley_data, alpha=0.4)
    
    # Convert back to array for rib and angle markers
    segmented_img = np.array(working_image)
    
    # Draw rib markers
    if 'inspiration' in results and 'positions' in results['inspiration']:
        for i, pos in enumerate(results['inspiration']['positions']):
            cv2.circle(segmented_img, (int(pos[0]), int(pos[1])), 5, (0, 255, 255), -1)
            cv2.putText(segmented_img, f'R{i+1}', 
                       (int(pos[0])+10, int(pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw costophrenic angle markers
    if 'costophrenic_angles' in results and 'angles' in results['costophrenic_angles']:
        for angle_data in results['costophrenic_angles']['angles']:
            pos = angle_data['position']
            color = (0, 255, 0) if angle_data['sharp'] else (255, 165, 0)
            cv2.circle(segmented_img, pos, 8, color, 2)
            if angle_data.get('angle'):
                label = f"{angle_data['angle']:.0f}°"
                cv2.putText(segmented_img, label,
                           (pos[0]+15, pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return Image.fromarray(segmented_img)

def create_comparison_figure(original, segmented, measurements_img):
    """Create comparison figure with all visualizations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    axes[1].imshow(segmented)
    axes[1].set_title('Segmentation Overlay')
    axes[1].axis('off')
    
    axes[2].imshow(measurements_img)
    axes[2].set_title('Measurements')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig