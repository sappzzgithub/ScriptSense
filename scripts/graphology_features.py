import cv2
import numpy as np
from PIL import Image

def map_to_behavior(feature, value):
    mapping = {
        "Letter Size": {
            "Large": "Likes being noticed, stands out in a crowd",
            "Small": "Introspective, not seeking attention, modest",
            "Medium": "Adaptable, fits into a crowd, practical, balanced",
            "Unknown": "Insufficient data to determine"
        },
        "Letter Slant": {
            "Right": "Sociable, responsive, interested in others, friendly",
            "Left": "Reserved, observant, self-reliant, non-intrusive",
            "Vertical": "Practical, independent, controlled, self-sufficient",
            "Unknown": "Insufficient data to determine"
        },
        "Pen Pressure": {
            "Light": "Can endure traumatic experiences without being seriously affected. Emotional experiences do not make a lasting impression",
            "Medium": "Balanced emotional state",
            "Heavy": "Have very deep and enduring feelings and feel situations intensely",
            "Unknown": "Insufficient data to determine"
        },
        "Baseline": {
            "Rising": "Optimistic, upbeat, positive attitude, ambitious and hopeful",
            "Falling": "Tired, overwhelmed, pessimistic, not hopeful",
            "Straight": "Determined, stays on track, self-motivated, controls emotions, reliable, steady",
            "Unknown": "Insufficient data to determine"
        }
    }
    return mapping.get(feature, {}).get(value, "Unknown")

def detect_letter_size(binary_img):
    # Add morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    
    heights = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:  # Reduced minimum area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            heights.append(h)
    
    if not heights:
        return "Unknown"
        
    avg_height = np.mean(heights)
    img_height = binary_img.shape[0]
    
    # Scale the thresholds based on image height
    if avg_height < img_height/10:
        return "Small"
    elif avg_height < img_height/5:
        return "Medium"
    else:
        return "Large"

def detect_letter_slant(binary_img):
    # Improved edge detection with adaptive thresholding
    edges = cv2.Canny(binary_img, 30, 100)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
    if lines is None:
        return "Vertical"
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > 5:  # Filter out near-vertical lines
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if -80 < angle < 80:  # Wider angle range
                angles.append(angle)
    
    if not angles:
        return "Vertical"
        
    avg_angle = np.mean(angles)
    if avg_angle < -10:  # More tolerant thresholds
        return "Left"
    elif avg_angle > 10:
        return "Right"
    else:
        return "Vertical"

def detect_pen_pressure(gray_img):
    # Use adaptive thresholding to better handle varying backgrounds
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    mean_intensity = np.mean(gray_img[thresh == 255])
    
    if mean_intensity < 60:
        return "Heavy"
    elif mean_intensity < 140:
        return "Medium"
    else:
        return "Light"

def detect_baseline(binary_img):
    # Improved baseline detection with horizontal projections
    horizontal_proj = np.sum(binary_img, axis=1)
    threshold = np.max(horizontal_proj) * 0.2
    indices = np.where(horizontal_proj > threshold)[0]
    
    if len(indices) < 2:
        return "Unknown"
    
    try:
        slope, intercept = np.polyfit(indices, np.arange(len(indices)), 1)
    except:
        return "Unknown"
    
    if slope > 0.15:
        return "Rising"
    elif slope < -0.15:
        return "Falling"
    else:
        return "Straight"

def detect_word_spacing(binary_img):
    # Vertical projection with improved gap detection
    vertical_proj = np.sum(binary_img, axis=0)
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, val in enumerate(vertical_proj):
        if val < 5 and not in_gap:
            in_gap = True
            gap_start = i
        elif val >= 5 and in_gap:
            in_gap = False
            gaps.append(i - gap_start)
    
    if not gaps:
        return "Unknown"
    
    avg_gap = np.mean(gaps)
    img_width = binary_img.shape[1]
    
    if avg_gap < img_width/50:
        return "Narrow"
    elif avg_gap < img_width/20:
        return "Normal"
    else:
        return "Wide"

def extract_graphology_features(image_path):
    try:
        img = Image.open(image_path).convert("L")
        gray = np.array(img)
        
        # Improved preprocessing with adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Additional cleaning
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        features = {
            "Letter Size": detect_letter_size(binary),
            "Letter Slant": detect_letter_slant(binary),
            "Pen Pressure": detect_pen_pressure(gray),
            "Baseline": detect_baseline(binary),
        }
        
        output = []
        for feature, value in features.items():
            behavior = map_to_behavior(feature, value)
            output.append({
                "Attribute": feature,
                "Writing Category": value,
                "Psychological Personality Behavior": behavior
            })
            
        return output
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return default unknown values if processing fails
        return [{
            "Attribute": feat,
            "Writing Category": "Unknown",
            "Psychological Personality Behavior": "Insufficient data to determine"
        } for feat in ["Letter Size", "Letter Slant", "Pen Pressure", "Baseline"]]