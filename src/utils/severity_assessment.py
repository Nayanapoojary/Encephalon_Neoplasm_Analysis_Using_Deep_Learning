"""
Tumor Severity Assessment and Clinical Consultation Module
"""

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

class TumorSeverityAssessment:
    """Comprehensive tumor severity and consultation system"""
    
    def __init__(self):
        # WHO Grading System adapted for imaging
        self.severity_criteria = {
            'size_based': {
                'grade_1': {'max_size': 2.0, 'label': 'Grade I - Low Risk', 'color': '#27ae60'},
                'grade_2': {'max_size': 5.0, 'label': 'Grade II - Moderate Risk', 'color': '#f39c12'},
                'grade_3': {'max_size': 8.0, 'label': 'Grade III - High Risk', 'color': '#e67e22'},
                'grade_4': {'max_size': float('inf'), 'label': 'Grade IV - Critical', 'color': '#e74c3c'}
            }
        }
        
        # Clinical guidelines
        self.consultation_protocols = self._load_consultation_protocols()
        self.treatment_timelines = self._load_treatment_timelines()
    
    def assess_severity(self, tumor_type, segmentation_metrics, classification_confidence):
        """
        Comprehensive severity assessment
        
        Parameters:
        - tumor_type: str (Glioma, Meningioma, Pituitary)
        - segmentation_metrics: dict with area_percentage, bounding_box, etc.
        - classification_confidence: float
        
        Returns:
        - dict with severity grade, risk level, clinical recommendations
        """
        
        if tumor_type == 'No Tumor':
            return {
                'severity_grade': 'N/A',
                'severity_score': 0,
                'risk_level': 'Normal',
                'urgency': 'Routine Follow-up',
                'color': '#27ae60',
                'description': 'No pathological findings detected',
                'tumor_type': tumor_type,
                'type_score': 0,
                'size_score': 0,
                'location_score': 0,
                'border_score': 0,
                'size_category': 'N/A',
                'area_percentage': 0,
                'classification_confidence': classification_confidence
            }
        
        # Calculate severity based on multiple factors
        area_percentage = segmentation_metrics.get('area_percentage', 0)
        has_segmentation = segmentation_metrics.get('tumor_detected', False)
        
        # Size-based assessment
        if has_segmentation and area_percentage > 0:
            severity_data = self._calculate_severity_score(
                tumor_type, 
                area_percentage, 
                segmentation_metrics,
                classification_confidence
            )
            severity_score = severity_data['total_score']
        else:
            # Classification-only assessment
            severity_data = self._classification_based_severity(
                tumor_type, 
                classification_confidence
            )
            severity_score = severity_data['total_score']
        
        # Determine grade
        grade_info = self._get_grade_from_score(severity_score)
        
        # Get clinical recommendations
        recommendations = self._get_clinical_recommendations(
            tumor_type, 
            grade_info, 
            has_segmentation
        )
        
        # Calculate urgency timeline
        timeline = self._calculate_consultation_timeline(
            tumor_type, 
            grade_info,
            classification_confidence
        )
        
        # Combine all information
        result = {
            'severity_grade': grade_info['label'],
            'severity_score': severity_score,
            'risk_level': grade_info['risk_level'],
            'urgency': grade_info['urgency'],
            'color': grade_info['color'],
            'description': grade_info['description'],
            'size_category': self._get_size_category(area_percentage),
            'consultation_timeline': timeline,
            'clinical_recommendations': recommendations,
            'requires_biopsy': self._requires_biopsy(tumor_type, severity_score),
            'imaging_followup': self._get_imaging_schedule(grade_info),
            'specialist_referral': self._get_specialist_type(tumor_type, grade_info),
            
            # Add detailed breakdown for visualization
            'tumor_type': tumor_type,
            'type_score': severity_data.get('type_score', 0),
            'size_score': severity_data.get('size_score', 0),
            'location_score': severity_data.get('location_score', 0),
            'border_score': severity_data.get('border_score', 0),
            'area_percentage': area_percentage,
            'classification_confidence': classification_confidence,
            'malignancy_potential': severity_data.get('malignancy', 'Unknown'),
            'typical_growth_rate': severity_data.get('growth_rate', 'Unknown'),
            'typical_who_grade': severity_data.get('who_grade', 'N/A'),
            'location_assessment': severity_data.get('location', 'Central'),
            'location_risk': severity_data.get('location_risk', 'Moderate Risk'),
            'border_quality': severity_data.get('border_quality', 'Moderately Defined')
        }
        
        return result
    
    def _calculate_severity_score(self, tumor_type, area_percentage, seg_metrics, confidence):
        """Calculate comprehensive severity score with detailed breakdown"""
        
        # Tumor type scoring (0-10 scale)
        type_scores = {
            'Glioma': {
                'score': 7.5,
                'malignancy': 'Potentially malignant (varies by grade)',
                'growth_rate': 'Variable (slow to aggressive)',
                'who_grade': 'I-IV'
            },
            'Meningioma': {
                'score': 3.0,
                'malignancy': 'Usually benign (90%)',
                'growth_rate': 'Slow',
                'who_grade': 'I'
            },
            'Pituitary': {
                'score': 2.5,
                'malignancy': 'Usually benign',
                'growth_rate': 'Very Slow',
                'who_grade': 'I'
            }
        }
        
        type_info = type_scores.get(tumor_type, {'score': 5.0, 'malignancy': 'Unknown', 'growth_rate': 'Unknown', 'who_grade': 'N/A'})
        type_score = type_info['score']
        
        # Size scoring (0-10 scale)
        if area_percentage < 0.5:
            size_score = 0.5
        elif area_percentage < 1.0:
            size_score = 1.5
        elif area_percentage < 3.0:
            size_score = 3.0
        elif area_percentage < 7.0:
            size_score = 5.5
        elif area_percentage < 12.0:
            size_score = 7.5
        else:
            size_score = 9.5
        
        # Location scoring (0-10 scale) - based on typical locations
        # This is a simplified model; in practice would use actual coordinates
        location_score = 6.0  # Default moderate risk
        location = 'Central'
        location_risk = 'Moderate Risk'
        
        # Border characteristics (0-10 scale)
        perimeter = seg_metrics.get('perimeter', 0)
        area = seg_metrics.get('area_pixels', 1)
        
        if perimeter > 0 and area > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.8:
                border_score = 2.0
                border_quality = 'Well-Defined'
            elif circularity > 0.6:
                border_score = 4.0
                border_quality = 'Moderately Defined'
            else:
                border_score = 7.0
                border_quality = 'Poorly Defined (Irregular)'
        else:
            border_score = 4.0
            border_quality = 'Moderately Defined'
        
        # Calculate weighted total score (0-100 scale)
        total_score = (
            type_score * 3.0 +      # 30% weight
            size_score * 1.5 +      # 15% weight
            location_score * 6.0 +  # 60% weight... wait, let me fix this
            border_score * 4.0      # 40% weight
        ) / 10 * 10  # Normalize to 100
        
        # Actually, let's use proper weighting
        total_score = (
            type_score * 0.30 +     # 30% weight
            size_score * 0.15 +     # 15% weight  
            location_score * 0.35 + # 35% weight
            border_score * 0.20     # 20% weight
        ) * 10  # Scale to 100
        
        # Add confidence bonus
        confidence_bonus = (confidence / 100) * 5
        total_score = min(total_score + confidence_bonus, 100)
        
        return {
            'total_score': total_score,
            'type_score': type_score,
            'size_score': size_score,
            'location_score': location_score,
            'border_score': border_score,
            'malignancy': type_info['malignancy'],
            'growth_rate': type_info['growth_rate'],
            'who_grade': type_info['who_grade'],
            'location': location,
            'location_risk': location_risk,
            'border_quality': border_quality
        }
    
    def _classification_based_severity(self, tumor_type, confidence):
        """Severity when segmentation not available"""
        
        type_scores = {
            'Glioma': {
                'base': 65,
                'malignancy': 'Potentially malignant',
                'growth_rate': 'Variable',
                'who_grade': 'I-IV'
            },
            'Meningioma': {
                'base': 35,
                'malignancy': 'Usually benign (90%)',
                'growth_rate': 'Slow',
                'who_grade': 'I'
            },
            'Pituitary': {
                'base': 30,
                'malignancy': 'Usually benign',
                'growth_rate': 'Very Slow',
                'who_grade': 'I'
            }
        }
        
        type_info = type_scores.get(tumor_type, {'base': 50, 'malignancy': 'Unknown', 'growth_rate': 'Unknown', 'who_grade': 'N/A'})
        base = type_info['base']
        confidence_factor = (confidence / 100) * 20
        
        total_score = min(base + confidence_factor, 85)
        
        # Estimate component scores
        type_score = 6.0 if tumor_type == 'Glioma' else 3.0
        size_score = 4.0  # Unknown, assume moderate
        location_score = 5.0  # Unknown, assume moderate
        border_score = 4.0  # Unknown, assume moderate
        
        return {
            'total_score': total_score,
            'type_score': type_score,
            'size_score': size_score,
            'location_score': location_score,
            'border_score': border_score,
            'malignancy': type_info['malignancy'],
            'growth_rate': type_info['growth_rate'],
            'who_grade': type_info['who_grade'],
            'location': 'Unknown (Multi-view MRI recommended)',
            'location_risk': 'Cannot assess without segmentation',
            'border_quality': 'Cannot assess without segmentation'
        }
    
    def _get_grade_from_score(self, score):
        """Convert severity score to clinical grade"""
        
        if score < 30:
            return {
                'label': 'Grade I - Low Risk',
                'risk_level': 'Low',
                'urgency': 'Routine Follow-up',
                'color': '#27ae60',
                'description': 'Small, well-defined tumor. Slow growth expected.'
            }
        elif score < 50:
            return {
                'label': 'Grade II - Moderate Risk',
                'risk_level': 'Moderate',
                'urgency': 'Consultation within 2-4 weeks',
                'color': '#f39c12',
                'description': 'Moderate-sized tumor. Active monitoring recommended.'
            }
        elif score < 70:
            return {
                'label': 'Grade III - High Risk',
                'risk_level': 'High',
                'urgency': 'Urgent consultation within 1 week',
                'color': '#e67e22',
                'description': 'Large or aggressive-appearing tumor. Prompt intervention may be needed.'
            }
        else:
            return {
                'label': 'Grade IV - Critical',
                'risk_level': 'Critical',
                'urgency': 'IMMEDIATE consultation (24-48 hours)',
                'color': '#e74c3c',
                'description': 'Extensive tumor or highly aggressive features. Immediate medical attention required.'
            }
    
    def _get_size_category(self, area_percentage):
        """Categorize tumor by size"""
        
        if area_percentage == 0:
            return 'Not Detected'
        elif area_percentage < 1.0:
            return 'Very Small'
        elif area_percentage < 3.0:
            return 'Small'
        elif area_percentage < 7.0:
            return 'Moderate'
        elif area_percentage < 12.0:
            return 'Large'
        else:
            return 'Very Large'
    
    def _get_clinical_recommendations(self, tumor_type, grade_info, has_segmentation):
        """Generate clinical recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'diagnostic_tests': [],
            'imaging_followup': [],
            'specialist_consultation': [],
            'lifestyle_modifications': []
        }
        
        risk = grade_info['risk_level']
        
        # Immediate actions based on risk
        if risk == 'Critical':
            recommendations['immediate_actions'] = [
                'Emergency neurosurgical consultation',
                'Admit for observation if symptomatic',
                'Baseline neurological examination',
                'Steroid therapy consideration for edema'
            ]
        elif risk == 'High':
            recommendations['immediate_actions'] = [
                'Schedule neurosurgical consultation within 7 days',
                'Complete neurological assessment',
                'Symptom monitoring protocol'
            ]
        elif risk == 'Moderate':
            recommendations['immediate_actions'] = [
                'Neurology consultation within 2-4 weeks',
                'Document any neurological symptoms',
                'Patient education materials'
            ]
        else:
            recommendations['immediate_actions'] = [
                'Routine follow-up in 3-6 months',
                'Monitor for new symptoms',
                'Maintain imaging records'
            ]
        
        # Diagnostic tests
        recommendations['diagnostic_tests'] = [
            'Contrast-enhanced MRI (multiple views)',
            'Complete blood count (CBC)',
            'Comprehensive metabolic panel'
        ]
        
        if tumor_type == 'Glioma':
            recommendations['diagnostic_tests'].extend([
                'MR Spectroscopy',
                'Perfusion MRI',
                'Consider PET scan for grading'
            ])
        elif tumor_type == 'Pituitary':
            recommendations['diagnostic_tests'].extend([
                'Hormonal panel (ACTH, TSH, Prolactin, GH)',
                'Visual field testing',
                'Pituitary function tests'
            ])
        elif tumor_type == 'Meningioma':
            recommendations['diagnostic_tests'].extend([
                'Ophthalmologic examination',
                'Hearing tests if near temporal bone'
            ])
        
        # Imaging follow-up schedule
        if risk == 'Critical':
            recommendations['imaging_followup'] = [
                'MRI in 1 month post-initial consultation',
                'Then every 2-3 months for first year',
                'Annual MRI thereafter if stable'
            ]
        elif risk == 'High':
            recommendations['imaging_followup'] = [
                'MRI in 3 months',
                'Then every 6 months for 2 years',
                'Annual thereafter'
            ]
        else:
            recommendations['imaging_followup'] = [
                'MRI in 6-12 months',
                'Annual MRI for 3-5 years',
                'Adjust based on findings'
            ]
        
        # Specialist referrals
        if tumor_type == 'Glioma':
            recommendations['specialist_consultation'] = [
                'Neurosurgeon (primary)',
                'Neuro-oncologist',
                'Radiation oncologist'
            ]
        elif tumor_type == 'Pituitary':
            recommendations['specialist_consultation'] = [
                'Pituitary neurosurgeon',
                'Endocrinologist',
                'Ophthalmologist (if visual symptoms)'
            ]
        else:
            recommendations['specialist_consultation'] = [
                'Neurosurgeon',
                'Neurologist',
                'Radiation oncologist (if indicated)'
            ]
        
        # Lifestyle modifications
        recommendations['lifestyle_modifications'] = [
            'Avoid activities with high head injury risk',
            'Monitor and report new headaches or vision changes',
            'Maintain regular sleep schedule',
            'Stress management techniques',
            'Document seizures if they occur'
        ]
        
        return recommendations
    
    def _calculate_consultation_timeline(self, tumor_type, grade_info, confidence):
        """Calculate when consultations should occur"""
        
        now = datetime.now()
        timeline = {}
        
        risk = grade_info['risk_level']
        
        if risk == 'Critical':
            timeline['neurosurgeon'] = now + timedelta(hours=48)
            timeline['imaging'] = now + timedelta(days=7)
            timeline['biopsy'] = now + timedelta(days=14)
        elif risk == 'High':
            timeline['neurosurgeon'] = now + timedelta(days=7)
            timeline['imaging'] = now + timedelta(days=30)
            timeline['biopsy'] = now + timedelta(days=60)
        elif risk == 'Moderate':
            timeline['neurosurgeon'] = now + timedelta(days=21)
            timeline['imaging'] = now + timedelta(days=90)
            timeline['biopsy'] = 'As needed'
        else:
            timeline['neurosurgeon'] = now + timedelta(days=90)
            timeline['imaging'] = now + timedelta(days=180)
            timeline['biopsy'] = 'As needed'
        
        return {k: v.strftime('%B %d, %Y') if isinstance(v, datetime) else v 
                for k, v in timeline.items()}
    
    def _requires_biopsy(self, tumor_type, severity_score):
        """Determine if biopsy is recommended"""
        
        if tumor_type == 'Glioma' and severity_score > 50:
            return {
                'recommended': True,
                'urgency': 'High',
                'reason': 'Glioma requires tissue diagnosis for grading and treatment planning'
            }
        elif severity_score > 70:
            return {
                'recommended': True,
                'urgency': 'Moderate',
                'reason': 'Large tumor size warrants tissue confirmation'
            }
        else:
            return {
                'recommended': False,
                'urgency': 'Low',
                'reason': 'May defer pending clinical correlation and imaging follow-up'
            }
    
    def _get_imaging_schedule(self, grade_info):
        """Get recommended imaging follow-up schedule"""
        
        risk = grade_info['risk_level']
        
        schedules = {
            'Critical': 'MRI every 2-3 months for first year',
            'High': 'MRI every 3-6 months',
            'Moderate': 'MRI every 6-12 months',
            'Low': 'MRI annually for 3-5 years'
        }
        
        return schedules.get(risk, 'As clinically indicated')
    
    def _get_specialist_type(self, tumor_type, grade_info):
        """Primary specialist to consult"""
        
        specialists = {
            'Glioma': 'Neurosurgeon + Neuro-oncologist',
            'Meningioma': 'Neurosurgeon',
            'Pituitary': 'Pituitary Neurosurgeon + Endocrinologist'
        }
        
        return specialists.get(tumor_type, 'Neurosurgeon')
    
    def _load_consultation_protocols(self):
        """Load clinical consultation protocols"""
        return {
            'glioma': {
                'WHO_grade_I': 'Surgical resection consideration',
                'WHO_grade_II': 'Surgery + possible adjuvant therapy',
                'WHO_grade_III': 'Surgery + radiation + chemotherapy',
                'WHO_grade_IV': 'Aggressive multimodal therapy'
            }
        }
    
    def _load_treatment_timelines(self):
        """Load treatment timeline guidelines"""
        return {
            'preoperative': '2-4 weeks',
            'postoperative_imaging': '24-48 hours',
            'radiation_start': '2-6 weeks post-surgery',
            'chemotherapy_start': 'Concurrent with radiation'
        }
    
    def generate_severity_report(self, assessment):
        """Generate formatted severity report"""
        
        # Get values with safe defaults
        severity_grade = assessment.get('severity_grade', 'N/A')
        risk_level = assessment.get('risk_level', 'Unknown')
        urgency = assessment.get('urgency', 'N/A')
        size_category = assessment.get('size_category', 'N/A')
        severity_score = assessment.get('severity_score', 0)
        
        timeline = assessment.get('consultation_timeline', {})
        neurosurgeon = timeline.get('neurosurgeon', 'N/A')
        imaging = timeline.get('imaging', 'N/A')
        biopsy = timeline.get('biopsy', 'N/A')
        
        requires_biopsy = assessment.get('requires_biopsy', {})
        biopsy_rec = 'Yes' if requires_biopsy.get('recommended', False) else 'No'
        biopsy_urgency = requires_biopsy.get('urgency', 'N/A')
        biopsy_reason = requires_biopsy.get('reason', 'N/A')
        
        imaging_followup = assessment.get('imaging_followup', 'N/A')
        specialist_referral = assessment.get('specialist_referral', 'N/A')
        
        recommendations = assessment.get('clinical_recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        diagnostic_tests = recommendations.get('diagnostic_tests', [])
        specialist_consultation = recommendations.get('specialist_consultation', [])
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    TUMOR SEVERITY ASSESSMENT                        ║
╚════════════════════════════════════════════════════════════════════╝

SEVERITY GRADE: {severity_grade}
RISK LEVEL: {risk_level}
URGENCY: {urgency}

SIZE CATEGORY: {size_category}
SEVERITY SCORE: {severity_score:.1f}/100

CONSULTATION TIMELINE:
  • Neurosurgeon: {neurosurgeon}
  • Follow-up Imaging: {imaging}
  • Biopsy: {biopsy}

BIOPSY RECOMMENDATION:
  • Required: {biopsy_rec}
  • Urgency: {biopsy_urgency}
  • Reason: {biopsy_reason}

IMAGING SCHEDULE: {imaging_followup}
PRIMARY SPECIALIST: {specialist_referral}

═══════════════════════════════════════════════════════════════════════
CLINICAL RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════

IMMEDIATE ACTIONS:
"""
        for action in immediate_actions:
            report += f"  ✓ {action}\n"
        
        report += "\nDIAGNOSTIC TESTS RECOMMENDED:\n"
        for test in diagnostic_tests:
            report += f"  • {test}\n"
        
        report += "\nSPECIALIST CONSULTATIONS:\n"
        for specialist in specialist_consultation:
            report += f"  • {specialist}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def generate_severity_visualization(self, severity_data, save_path='severity_report.png'):
        """
        Generate a comprehensive severity assessment visualization
        
        Args:
            severity_data: Dictionary containing severity assessment results
            save_path: Path to save the visualization
        """
        # Create figure with specific layout
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('TUMOR SEVERITY ASSESSMENT REPORT', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.5, 0.3], 
                              width_ratios=[1, 1.2], hspace=0.3, wspace=0.3,
                              left=0.08, right=0.95, top=0.92, bottom=0.05)
        
        # ====================================================================
        # 1. OVERALL SEVERITY SCORE (Top Left)
        # ====================================================================
        ax_score = fig.add_subplot(gs[0, 0])
        ax_score.set_xlim(0, 1)
        ax_score.set_ylim(0, 1)
        ax_score.axis('off')
        
        # Add title
        ax_score.text(0.5, 0.95, 'Overall Severity Score', 
                      ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Draw circle with severity score
        circle = mpatches.Circle((0.5, 0.45), 0.35, 
                                facecolor='#FFE5B4', 
                                edgecolor='#F5A623', 
                                linewidth=3)
        ax_score.add_patch(circle)
        
        # Severity score text
        severity_score = severity_data.get('severity_score', 0)
        ax_score.text(0.5, 0.55, f'{severity_score:.1f}/100', 
                      ha='center', va='center', 
                      fontsize=42, fontweight='bold', color='#D4860B')
        
        # Severity grade - extract just the grade part
        severity_grade = severity_data.get('severity_grade', 'N/A')
        # Extract just "MODERATE" or "LOW" etc from "Grade II - Moderate Risk"
        if '-' in severity_grade:
            grade_display = severity_grade.split('-')[1].strip().split()[0].upper()
        else:
            grade_display = severity_grade
        
        ax_score.text(0.5, 0.32, grade_display, 
                      ha='center', va='center', 
                      fontsize=22, fontweight='bold', color='#D4860B')
        
        # Urgency text
        urgency = severity_data.get('urgency', 'N/A')
        ax_score.text(0.5, 0.15, urgency, 
                      ha='center', va='center', 
                      fontsize=11, style='italic', color='#333333')
        
        # ====================================================================
        # 2. COMPONENT ANALYSIS (Top Right)
        # ====================================================================
        ax_components = fig.add_subplot(gs[0, 1])
        
        # Get component scores
        type_score = severity_data.get('type_score', 0)
        size_score = severity_data.get('size_score', 0)
        location_score = severity_data.get('location_score', 0)
        border_score = severity_data.get('border_score', 0)
        
        components = ['Type', 'Size', 'Location', 'Borders']
        scores = [type_score, size_score, location_score, border_score]
        
        # Color mapping based on score
        colors = []
        for score in scores:
            if score <= 2.5:
                colors.append('#5DADE2')  # Blue - Low
            elif score <= 5:
                colors.append('#48C9B0')  # Teal - Moderate
            elif score <= 7.5:
                colors.append('#F39C12')  # Orange - High
            else:
                colors.append('#E74C3C')  # Red - Critical
        
        # Create horizontal bar chart
        y_pos = np.arange(len(components))
        bars = ax_components.barh(y_pos, scores, color=colors, edgecolor='#333333', linewidth=1.5)
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax_components.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                              f'{score:.1f}', 
                              ha='left', va='center', fontweight='bold', fontsize=11)
        
        ax_components.set_yticks(y_pos)
        ax_components.set_yticklabels(components, fontsize=12)
        ax_components.set_xlabel('Severity Score', fontsize=12, fontweight='bold')
        ax_components.set_title('Component Analysis', fontsize=16, fontweight='bold', pad=10)
        ax_components.set_xlim(0, 10)
        ax_components.grid(axis='x', alpha=0.3, linestyle='--')
        ax_components.spines['top'].set_visible(False)
        ax_components.spines['right'].set_visible(False)
        
        # ====================================================================
        # 3. RISK FACTORS BREAKDOWN (Bottom Left)
        # ====================================================================
        ax_risk = fig.add_subplot(gs[1, 0])
        ax_risk.axis('off')
        
        # Create text box background
        risk_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='#FFF8DC', 
                                  edgecolor='#333333', 
                                  linewidth=2,
                                  transform=ax_risk.transAxes)
        ax_risk.add_patch(risk_box)
        
        # Title
        ax_risk.text(0.5, 0.95, 'RISK FACTORS BREAKDOWN', 
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=ax_risk.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333333'))
        
        # Separator line
        ax_risk.plot([0.05, 0.95], [0.90, 0.90], 'k-', linewidth=1.5, transform=ax_risk.transAxes)
        
        # Risk factors text
        y_start = 0.85
        line_height = 0.08
        
        # Tumor Type
        tumor_type = severity_data.get('tumor_type', 'N/A')
        malignancy = severity_data.get('malignancy_potential', 'Unknown')
        growth_rate = severity_data.get('typical_growth_rate', 'Unknown')
        who_grade = severity_data.get('typical_who_grade', 'N/A')
        
        ax_risk.text(0.05, y_start, '• Tumor Type', fontsize=11, fontweight='bold', 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.04, f'Value: {tumor_type}', fontsize=9, 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.07, f'Score: {type_score:.1f}/10', fontsize=9, 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.10, f'typical_grades: {who_grade}', 
                    fontsize=9, transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.13, f'malignancy: {malignancy}', fontsize=9, 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.16, f'growth_rate: {growth_rate}', fontsize=9, 
                    transform=ax_risk.transAxes)
        
        y_start -= 0.24
        
        # Size
        size_category = severity_data.get('size_category', 'N/A')
        area_pct = severity_data.get('area_percentage', 0)
        
        ax_risk.text(0.05, y_start, '• Size', fontsize=11, fontweight='bold', 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.04, f'Value: {area_pct:.2f}% ({size_category})', 
                    fontsize=9, transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.07, f'Score: {size_score:.1f}/10', fontsize=9, 
                    transform=ax_risk.transAxes)
        
        y_start -= 0.14
        
        # Location
        location = severity_data.get('location_assessment', 'Unknown')
        location_risk = severity_data.get('location_risk', 'Unknown')
        
        ax_risk.text(0.05, y_start, '• Location', fontsize=11, fontweight='bold', 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.04, f'Value: {location} ({location_risk})', 
                    fontsize=9, transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.07, f'Score: {location_score:.1f}/10', fontsize=9, 
                    transform=ax_risk.transAxes)
        
        y_start -= 0.14
        
        # Border Characteristics
        border_quality = severity_data.get('border_quality', 'Unknown')
        
        ax_risk.text(0.05, y_start, '• Border Characteristics', fontsize=11, fontweight='bold', 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.04, f'Value: {border_quality}', fontsize=9, 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.07, f'Score: {border_score:.1f}/10', fontsize=9, 
                    transform=ax_risk.transAxes)
        
        y_start -= 0.14
        
        # Classification Confidence
        confidence = severity_data.get('classification_confidence', 0)
        ax_risk.text(0.05, y_start, '• Classification Confidence', fontsize=11, fontweight='bold', 
                    transform=ax_risk.transAxes)
        ax_risk.text(0.07, y_start - 0.04, f'Value: {confidence:.1f}%', fontsize=9, 
                    transform=ax_risk.transAxes)
        
        # ====================================================================
        # 4. CLINICAL RECOMMENDATIONS (Bottom Right)
        # ====================================================================
        ax_clinical = fig.add_subplot(gs[1, 1])
        ax_clinical.axis('off')
        
        # Create text box background
        clinical_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, 
                                      boxstyle="round,pad=0.02", 
                                      facecolor='#E8F4F8', 
                                      edgecolor='#333333', 
                                      linewidth=2,
                                      transform=ax_clinical.transAxes)
        ax_clinical.add_patch(clinical_box)
        
        # Title
        ax_clinical.text(0.5, 0.95, 'CLINICAL RECOMMENDATIONS', 
                        ha='center', va='top', fontsize=14, fontweight='bold',
                        transform=ax_clinical.transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333333'))
        
        # Separator line
        ax_clinical.plot([0.05, 0.95], [0.90, 0.90], 'k-', linewidth=1.5, 
                        transform=ax_clinical.transAxes)
        
        # Get recommendations
        recommendations = severity_data.get('clinical_recommendations', {})
        
        # Display recommendations as bullet points
        y_pos = 0.85
        line_spacing = 0.06
        
        # Tumor-specific note
        tumor_info = f"{tumor_type} detected - "
        if tumor_type == "Meningioma":
            tumor_info += "Usually benign, but monitoring needed"
        elif tumor_type == "Glioma":
            tumor_info += "Requires immediate attention and treatment planning"
        elif tumor_type == "Pituitary":
            tumor_info += "May affect hormone levels, endocrine evaluation needed"
        elif tumor_type == "No Tumor":
            tumor_info += "No abnormalities detected"
        else:
            tumor_info += "Comprehensive evaluation required"
        
        ax_clinical.text(0.06, y_pos, f"☐ {tumor_info}", 
                        fontsize=9, transform=ax_clinical.transAxes, 
                        wrap=True, va='top')
        y_pos -= line_spacing * 1.5
        
        # Key recommendations
        key_recs = [
            "Serial MRI scans to track growth rate",
            "Conservative management with regular follow-up",
            "Upload multiple MRI views (Axial, Coronal, Sagittal) for comprehensive assessment",
            "Clinical correlation with symptoms and neurological exam essential"
        ]
        
        for rec in key_recs:
            if y_pos > 0.15:  # Keep within bounds
                ax_clinical.text(0.06, y_pos, f"☐ {rec}", 
                               fontsize=9, transform=ax_clinical.transAxes,
                               wrap=True, va='top')
                y_pos -= line_spacing * 1.2
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path


# Example usage
if __name__ == "__main__":
    assessor = TumorSeverityAssessment()
    
    # Example 1: Large Glioma
    assessment = assessor.assess_severity(
        tumor_type='Glioma',
        segmentation_metrics={
            'tumor_detected': True,
            'area_percentage': 5.2,
            'area_pixels': 2600,
            'perimeter': 350.5,
            'centroid': {'x': 100, 'y': 120}
        },
        classification_confidence=97.5
    )
    
    print(assessor.generate_severity_report(assessment))
    
    # Generate visualization
    assessor.generate_severity_visualization(assessment, 'test_severity_report.png')
    print("\n✓ Severity visualization saved to test_severity_report.png")
    
    # Example 2: Small Meningioma
    assessment2 = assessor.assess_severity(
        tumor_type='Meningioma',
        segmentation_metrics={
            'tumor_detected': True,
            'area_percentage': 1.8,
            'area_pixels': 900,
            'perimeter': 120.0,
            'centroid': {'x': 150, 'y': 80}
        },
        classification_confidence=99.2
    )
    
    print(assessor.generate_severity_report(assessment2))
    assessor.generate_severity_visualization(assessment2, 'test_severity_report_meningioma.png')
    print("✓ Meningioma severity visualization saved")