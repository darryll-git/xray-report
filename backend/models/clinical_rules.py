# Enhanced clinical thresholds with more conservative values
CLINICAL_THRESHOLDS = {
    # Critical/Urgent findings - higher threshold for specificity
    'Pneumothorax': 0.80,
    'Tension Pneumothorax': 0.85,
    'Massive Pleural Effusion': 0.75,
    'Large Opacity': 0.70,
    
    # Important findings - moderate threshold
    'Pleural Effusion': 0.65,
    'Consolidation': 0.70,
    'Pneumonia': 0.75,
    'Mass': 0.80,
    'Nodule': 0.70,
    'Cardiomegaly': 0.60,
    'Edema': 0.65,
    'Atelectasis': 0.65,
    
    # Less urgent findings - standard threshold
    'Infiltration': 0.60,
    'Emphysema': 0.65,
    'Fibrosis': 0.70,
    'Pleural Thickening': 0.60,
    'Lung Lesion': 0.65,
    'Lung Opacity': 0.60,
    'Enlarged Cardiomediastinum': 0.65,
    
    # Low clinical priority - higher threshold to reduce noise
    'Fracture': 0.75,
    'Hernia': 0.80,
    'Support Devices': 0.85
}

# Enhanced clinical priority levels
CLINICAL_PRIORITY = {
    'CRITICAL': ['Pneumothorax', 'Tension Pneumothorax', 'Massive Pleural Effusion'],
    'HIGH': ['Mass', 'Large Opacity', 'Consolidation', 'Pneumonia'],
    'MODERATE': ['Pleural Effusion', 'Nodule', 'Cardiomegaly', 'Edema', 'Atelectasis'],
    'LOW': ['Infiltration', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Fracture']
}

# Rule-based filtering patterns
CONTRADICTION_RULES = {
    # If pneumothorax present, suppress other lung findings unless very high confidence
    'pneumothorax_suppress': {
        'trigger': ['Pneumothorax'],
        'suppress': ['Infiltration', 'Lung Opacity', 'Edema'],
        'min_confidence': 0.85
    },
    # If consolidation present, suppress general infiltration
    'consolidation_suppress': {
        'trigger': ['Consolidation', 'Pneumonia'],
        'suppress': ['Infiltration'],
        'min_confidence': 0.80
    },
    # If mass present, suppress general nodule findings
    'mass_suppress': {
        'trigger': ['Mass'],
        'suppress': ['Nodule', 'Lung Lesion'],
        'min_confidence': 0.75
    },
    # Suppress redundant cardiac findings
    'cardiac_suppress': {
        'trigger': ['Cardiomegaly'],
        'suppress': ['Enlarged Cardiomediastinum'],
        'min_confidence': 0.70
    }
}

# Clinical significance patterns
LOW_CLINICAL_VALUE = [
    'Support Devices',  # Usually not pathological
    'Hernia',          # Often incidental
    'Pleural Thickening'  # Often chronic/stable
]

# Confidence language mapping
CONFIDENCE_LANGUAGE = {
    'CRITICAL': {
        'high': 'demonstrates',
        'moderate': 'strongly suggests', 
        'low': 'raises concern for'
    },
    'HIGH': {
        'high': 'shows evidence of',
        'moderate': 'suggests',
        'low': 'may indicate'
    },
    'MODERATE': {
        'high': 'indicates',
        'moderate': 'suggests possible',
        'low': 'cannot exclude'
    },
    'LOW': {
        'high': 'shows',
        'moderate': 'suggests',
        'low': 'possible'
    }
}

def get_pathology_priority(pathology):
    """Get clinical priority level for a pathology"""
    for priority, pathologies in CLINICAL_PRIORITY.items():
        if any(p.lower() in pathology.lower() for p in pathologies):
            return priority
    return 'LOW'

def apply_rule_based_filtering(findings):
    """Apply advanced rule-based filtering to suppress contradictory and low-value findings"""
    filtered_findings = {}
    suppressed_findings = {}
    
    # First pass: identify trigger conditions
    triggers_present = {}
    for rule_name, rule in CONTRADICTION_RULES.items():
        for trigger in rule['trigger']:
            for pathology, data in findings.items():
                prob = data if isinstance(data, (float, int)) else data.get('probability', 0)
                threshold = CLINICAL_THRESHOLDS.get(pathology, 0.5)
                if trigger.lower() in pathology.lower() and prob >= threshold:
                    triggers_present[rule_name] = {
                        'trigger_pathology': pathology,
                        'probability': prob
                    }
                    break
    
    # Second pass: apply suppression rules
    for pathology, data in findings.items():
        prob = data if isinstance(data, (float, int)) else data.get('probability', 0)
        threshold = CLINICAL_THRESHOLDS.get(pathology, 0.5)
        
        # Skip if below threshold
        if prob < threshold:
            continue
            
        # Check if this finding should be suppressed
        should_suppress = False
        suppression_reason = ""
        
        for rule_name, rule in CONTRADICTION_RULES.items():
            if rule_name in triggers_present:
                trigger_info = triggers_present[rule_name]
                
                # Check if current pathology should be suppressed
                for suppress_pattern in rule['suppress']:
                    if suppress_pattern.lower() in pathology.lower():
                        # Only suppress if trigger has higher confidence or meets minimum
                        if (trigger_info['probability'] >= rule['min_confidence'] and 
                            trigger_info['probability'] > prob):
                            should_suppress = True
                            suppression_reason = f"Suppressed due to {trigger_info['trigger_pathology']} presence"
                            break
                
                if should_suppress:
                    break
        
        # Apply low clinical value suppression
        if not should_suppress:
            for low_value_pattern in LOW_CLINICAL_VALUE:
                if low_value_pattern.lower() in pathology.lower() and prob < 0.85:
                    should_suppress = True
                    suppression_reason = "Low clinical significance - high threshold required"
                    break
        
        # Age-related suppression (for common aging changes)
        if not should_suppress:
            aging_patterns = ['Emphysema', 'Fibrosis', 'Pleural Thickening']
            for pattern in aging_patterns:
                if pattern.lower() in pathology.lower() and prob < 0.75:
                    should_suppress = True
                    suppression_reason = "Common age-related change - high threshold required"
                    break
        
        data_dict = data if isinstance(data, dict) else {'probability': prob}
        
        if should_suppress:
            suppressed_findings[pathology] = {
                **data_dict,
                'suppression_reason': suppression_reason
            }
        else:
            filtered_findings[pathology] = data_dict
    
    return filtered_findings, suppressed_findings

def create_clinical_summary(findings, view_analysis):
    """Create a clinically-focused summary with enhanced rule-based filtering"""
    
    # Apply rule-based filtering first
    filtered_findings, suppressed_findings = apply_rule_based_filtering(findings)
    
    # Categorize by clinical priority
    clinical_categories = {
        'CRITICAL': [],
        'HIGH': [],
        'MODERATE': [],
        'LOW': [],
        'NEGATIVE': [],
        'SUPPRESSED': []
    }
    
    for pathology, data in filtered_findings.items():
        priority = get_pathology_priority(pathology)
        threshold = CLINICAL_THRESHOLDS.get(pathology, 0.5)
        
        clinical_categories[priority].append({
            'pathology': pathology,
            'probability': data['probability'],
            'priority': priority,
            'threshold_used': threshold,
            'confidence_level': 'high' if data['probability'] > 0.8 else 'moderate' if data['probability'] > 0.65 else 'low'
        })
    
    # Add suppressed findings for transparency
    for pathology, data in suppressed_findings.items():
        clinical_categories['SUPPRESSED'].append({
            'pathology': pathology,
            'probability': data['probability'],
            'suppression_reason': data['suppression_reason']
        })
    
    # Add negative findings for context
    negative_findings = []
    for pathology, data in findings.items():
        prob = data if isinstance(data, (float, int)) else data.get('probability', 0)
        threshold = CLINICAL_THRESHOLDS.get(pathology, 0.5)
        if prob < threshold and pathology not in suppressed_findings:
            negative_findings.append({
                'pathology': pathology,
                'probability': prob
            })
    
    negative_findings.sort(key=lambda x: x['probability'], reverse=True)
    clinical_categories['NEGATIVE'] = negative_findings[:5]
    
    # Sort each category by probability
    for category in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']:
        clinical_categories[category].sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'view_analysis': view_analysis,
        'clinical_findings': clinical_categories,
        'total_positive_findings': sum(len(clinical_categories[cat]) for cat in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']),
        'total_suppressed': len(clinical_categories['SUPPRESSED']),
        'most_significant': clinical_categories['CRITICAL'][0] if clinical_categories['CRITICAL'] 
                          else clinical_categories['HIGH'][0] if clinical_categories['HIGH']
                          else clinical_categories['MODERATE'][0] if clinical_categories['MODERATE']
                          else None
    }
