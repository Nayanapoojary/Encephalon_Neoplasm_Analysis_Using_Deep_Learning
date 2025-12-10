"""
Enhanced MRI Image Validator
Validates if uploaded image is a valid brain MRI scan
"""

import numpy as np
from PIL import Image
import cv2
import os


class MRIValidator:
    """Enhanced MRI Image Validator with brain MRI detection"""
    
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz']
    MIN_SIZE = (50, 50)
    MAX_SIZE = (4096, 4096)
    VALID_MODES = ['L', 'RGB', 'RGBA', 'I', 'F']
    
    @staticmethod
    def validate_mri(image_path, strict=True):
        """
        Comprehensive MRI validation including brain MRI detection
        
        Args:
            image_path (str): Path to image file
            strict (bool): Enable strict validation
            
        Returns:
            tuple: (is_valid, message, metadata)
        """
        metadata = {}
        
        try:
            # 1. File existence check
            if not os.path.exists(image_path):
                return False, "❌ File does not exist", metadata
            
            # 2. File extension check
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in MRIValidator.SUPPORTED_FORMATS:
                return False, f"❌ Unsupported format: {ext}", metadata
            
            # 3. File size check
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return False, "❌ Empty file", metadata
            
            metadata['file_size'] = file_size
            metadata['extension'] = ext
            
            # 4. Image loading check
            try:
                img = Image.open(image_path)
                img_array = np.array(img)
                metadata['mode'] = img.mode
                metadata['format'] = img.format
                metadata['size'] = img.size
            except Exception as e:
                return False, f"❌ Cannot open image: {str(e)}", metadata
            
            # 5. Image size validation
            width, height = img.size
            
            if width < MRIValidator.MIN_SIZE[0] or height < MRIValidator.MIN_SIZE[1]:
                return False, f"❌ Image too small: {width}x{height}", metadata
            
            if width > MRIValidator.MAX_SIZE[0] or height > MRIValidator.MAX_SIZE[1]:
                return False, f"❌ Image too large: {width}x{height}", metadata
            
            # 6. Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                # Check if it's a color photo (high saturation)
                is_color_photo, color_msg = MRIValidator._check_color_photo(img_array)
                if is_color_photo:
                    return False, f"❌ {color_msg}", metadata
                
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            metadata['array_shape'] = img_gray.shape
            metadata['dtype'] = str(img_gray.dtype)
            
            # 7. Check for blank images
            if np.std(img_gray) < 1e-6:
                return False, "❌ Image appears to be blank", metadata
            
            # 8. MRI-specific validation
            is_mri, mri_msg, mri_score = MRIValidator._validate_mri_characteristics(img_gray)
            metadata['mri_score'] = mri_score
            metadata['mri_details'] = mri_msg
            
            if not is_mri:
                return False, f"❌ {mri_msg}", metadata
            
            # Store statistics
            metadata['min'] = float(np.min(img_gray))
            metadata['max'] = float(np.max(img_gray))
            metadata['mean'] = float(np.mean(img_gray))
            metadata['std'] = float(np.std(img_gray))
            
            # All checks passed
            return True, f"✅ Valid brain MRI (confidence: {mri_score:.1f}%)", metadata
            
        except Exception as e:
            return False, f"❌ Validation error: {str(e)}", metadata
    
    @staticmethod
    def _check_color_photo(img_array):
        """Check if image is a regular color photo (not medical imaging)"""
        
        # Convert to HSV to check saturation
        if img_array.max() > 1:
            img_normalized = (img_array / 255.0 * 255).astype(np.uint8)
        else:
            img_normalized = (img_array * 255).astype(np.uint8)
        
        hsv = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Medical images typically have low saturation
        mean_saturation = np.mean(saturation)
        high_sat_pixels = np.sum(saturation > 50) / saturation.size
        
        # Check for highly saturated colors (typical of photos)
        if mean_saturation > 40 and high_sat_pixels > 0.3:
            return True, "Detected color photograph - not a medical scan"
        
        # Check for skin tones (typical of portraits)
        # HSV ranges for skin tones
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 170, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        
        if skin_percentage > 0.2:
            return True, "Detected human portrait/photo - not a brain MRI"
        
        return False, "Color check passed"
    
    @staticmethod
    def _validate_mri_characteristics(img_gray):
        """
        Validate if grayscale image has MRI characteristics
        
        Returns:
            tuple: (is_valid, message, confidence_score)
        """
        score = 0.0
        reasons = []
        
        # 1. Intensity distribution check (MRI has specific histogram patterns)
        hist, bins = np.histogram(img_gray, bins=256, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        
        # MRI typically has a peak at low intensities (dark background)
        if hist_normalized[:50].sum() > 0.3:  # More than 30% dark pixels
            score += 25
            reasons.append("Dark background present")
        else:
            reasons.append("Missing typical MRI dark background")
        
        # 2. Check for brain-like structure (central bright region)
        height, width = img_gray.shape
        center_region = img_gray[height//4:3*height//4, width//4:3*width//4]
        edge_region_top = img_gray[0:height//8, :]
        edge_region_bottom = img_gray[7*height//8:, :]
        
        center_mean = np.mean(center_region)
        edge_mean = (np.mean(edge_region_top) + np.mean(edge_region_bottom)) / 2
        
        # Brain MRI: center should be brighter than edges
        if center_mean > edge_mean * 1.2:
            score += 30
            reasons.append("Central bright region detected")
        else:
            reasons.append("No central bright region (brain structure)")
        
        # 3. Contrast check (MRI has good tissue contrast)
        contrast = np.std(img_gray)
        if 30 < contrast < 80:
            score += 20
            reasons.append(f"Good tissue contrast (σ={contrast:.1f})")
        else:
            reasons.append(f"Unusual contrast for MRI (σ={contrast:.1f})")
        
        # 4. Edge detection (brain has characteristic edges)
        edges = cv2.Canny(img_gray.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if 0.05 < edge_density < 0.25:  # Typical range for brain MRI
            score += 15
            reasons.append("Appropriate edge density")
        else:
            reasons.append("Edge density outside MRI range")
        
        # 5. Circular/elliptical structure detection (skull/brain shape)
        # Use Hough Circle detection
        blurred = cv2.GaussianBlur(img_gray.astype(np.uint8), (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100,
            param1=50, 
            param2=30, 
            minRadius=50, 
            maxRadius=min(height, width)//2
        )
        
        if circles is not None:
            score += 10
            reasons.append("Circular structure detected (skull/brain)")
        else:
            reasons.append("No circular structure found")
        
        # Determine if valid MRI
        is_valid = score >= 50  # Need at least 50% confidence
        
        if is_valid:
            message = f"Brain MRI characteristics detected: {', '.join(reasons)}"
        else:
            message = f"Not a brain MRI. Issues: {', '.join(reasons)}"
        
        return is_valid, message, score


def validate_mri(image_path, strict=True):
    """Convenience function for backward compatibility"""
    is_valid, message, metadata = MRIValidator.validate_mri(image_path, strict)
    return is_valid, message