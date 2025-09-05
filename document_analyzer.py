import cv2
import numpy as np
import re
import logging
from config import HAS_TESSERACT, HAS_FRENCH

logger = logging.getLogger(__name__)

def detect_id_card_model(text):
    """Detect if ID card is old or new model based on text content"""
    text_upper = text.upper()
    
    # Indicators of new model (2021+)
    new_model_indicators = [
        'IDENTITY CARD',  # Bilingual format
        '/IDENTITY CARD', # Format with slash
        'GIVEN NAMES',    # English text presence
        'PLACE OF BIRTH', # English labels
        'DATE OF BIRTH',
        'NATIONALITY'
    ]
    
    # Indicators of old model (pre-2021)
    old_model_indicators = [
        'TAILLE',         # Height field (only on old cards)
        'PREFECTURE',     # Issued by prefecture
        'VALABLE JUSQU'   # Different expiry format
    ]
    
    new_score = sum(1 for indicator in new_model_indicators if indicator in text_upper)
    old_score = sum(1 for indicator in old_model_indicators if indicator in text_upper)
    
    # Decision logic
    if new_score >= 2:
        return "new", f"New model detected (score: {new_score}/6)"
    elif old_score >= 1:
        return "old", f"Old model detected (score: {old_score}/4)"
    else:
        return "new", "Assumed new model (default)"

def extract_info_old_model(text):
    """Information extraction for OLD French ID cards (pre-2021)"""
    # Normalize text and split into lines
    text = text.replace('|', ' ').replace('Ne:', ' ').replace('$', ' ')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    full_text = ' '.join(lines)
    
    logger.info(f"=== OLD MODEL EXTRACTION ===")
    logger.info(f"Lines detected: {len(lines)}")
    
    # 1. IDENTITY NUMBER - 12 digits for old cards
    identity_number = None
    twelve_digit_match = re.search(r'\b(\d{12})\b', full_text)
    if twelve_digit_match:
        identity_number = twelve_digit_match.group(1)
        logger.info(f"12-digit identity number found: {identity_number}")
    else:
        all_digits = re.findall(r'\d', full_text)
        if len(all_digits) >= 12:
            identity_number = ''.join(all_digits)[:12]
    
    # 2. SURNAME - French labels
    surname = None
    surname_patterns = ['Nom', 'NOM']
    for pattern in surname_patterns:
        match = re.search(rf'{pattern}\s*[:]\s*([A-Z][A-Za-z\s-]+)', full_text)
        if match:
            surname = match.group(1).strip()
            logger.info(f"Surname found: '{surname}'")
            break
    
    # 3. FIRST NAMES - French labels
    first_names = None
    name_patterns = ['Prénom\\(s\\)', 'Prénoms', 'PRÉNOM']
    for pattern in name_patterns:
        match = re.search(rf'{pattern}\s*[:]\s*([A-Za-zÀ-ÿ\s,]+)', full_text)
        if match:
            first_names = match.group(1).strip()
            logger.info(f"First names found: '{first_names}'")
            break
    
    # 4. SEX - French format
    sex = None
    sex_match = re.search(r'Sexe\s*[:]\s*([MF])', full_text)
    if sex_match:
        sex = sex_match.group(1)
        logger.info(f"Sex found: '{sex}'")
    
    # 5. DATE OF BIRTH - Traditional format
    date_of_birth = None
    dob_patterns = [
        r'\b(\d{2})\.(\d{2})\.(\d{4})\b',  # DD.MM.YYYY
        r'(\d{2})/(\d{2})/(\d{4})',       # DD/MM/YYYY
    ]
    
    for pattern in dob_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            day, month, year = match.groups()
            year_int = int(year)
            if 1920 <= year_int <= 2005:
                date_of_birth = f"{day}.{month}.{year}"
                logger.info(f"Date of birth found: '{date_of_birth}'")
                break
        if date_of_birth:
            break
    
    # 6. PLACE OF BIRTH - French format
    place_of_birth = None
    place_match = re.search(r'à \s+([A-Za-zÀ-ÿ\s\-\d]+)', full_text)
    if place_match:
        place_of_birth = place_match.group(1).strip()
        logger.info(f"Place of birth found: '{place_of_birth}'")
    
    # 7. HEIGHT - Only on old cards
    height = None
    height_match = re.search(r'Taille\s*[:]\s*(\d[,\.]\d{2}\s*m)', full_text, re.IGNORECASE)
    if height_match:
        height = height_match.group(1)
        logger.info(f"Height found: '{height}'")
    
    extracted_data = { 
        "Identity number": identity_number if identity_number else "Not found",
        "Surname": surname if surname else "Not found", 
        "First name(s)": first_names if first_names else "Not found", 
        "Sex": sex if sex else "Not found", 
        "Date of birth": date_of_birth if date_of_birth else "Not found", 
        "Place of birth": place_of_birth if place_of_birth else "Not found", 
        "Height": height if height else "Not found",
    }
    
    detected_count = sum(1 for v in extracted_data.values() if v != 'Not found')
    logger.info(f"OLD MODEL: Fields detected: {detected_count}/7")
    
    return extracted_data

def extract_info_new_model(text):
    """Information extraction for NEW French ID cards (2021+)"""
    # Normalize text and split into lines
    text = text.replace('|', ' ').replace('Ne:', ' ').replace('$', ' ')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    full_text = ' '.join(lines)
    
    logger.info(f"=== NEW MODEL EXTRACTION ===")
    logger.info(f"Lines detected: {len(lines)}")
    
    # 1. DOCUMENT NUMBER - Exactly 9 characters (new format)
    identity_number = None
    nine_char_patterns = [
        r'\b([A-Z0-9]{9})\b',
        r'\b([A-Z]\d{2}[A-Z]\d{2}[A-Z]{2}\d)\b',
        r'\b([A-Z]{1,3}\d{1,6}[A-Z]{1,3})\b'
    ]
    
    for pattern in nine_char_patterns:
        matches = re.findall(pattern, full_text)
        for match in matches:
            if len(match) == 9 and re.search(r'[A-Z]', match) and re.search(r'\d', match):
                identity_number = match
                logger.info(f"9-character document number found: {identity_number}")
                break
        if identity_number:
            break
    
    # 2. SURNAME - Bilingual format
    surname = None
    for i, line in enumerate(lines):
        if 'NOM' in line.upper() and ('SURNAME' in line.upper() or '/' in line):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 2:
                    surname = next_line
                    logger.info(f"Surname found: '{surname}'")
                    break
    
    # 3. FIRST NAMES - Bilingual format
    first_names = None
    for i, line in enumerate(lines):
        if 'PRENOMS' in line.upper() or 'GIVEN NAMES' in line.upper():
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and ',' in next_line:
                    first_names = next_line
                    logger.info(f"First names found: '{first_names}'")
                    break
    
    # 4. SEX - Improved detection for new format
    sex = None
    for line in lines:
        if 'SEXE' in line.upper() or 'SEX' in line.upper():
            sex_match = re.search(r'\b([FM])\b', line.upper())
            if sex_match:
                sex = sex_match.group(1)
                logger.info(f"Sex found: '{sex}'")
                break
    
    # 5. NATIONALITY - New cards include this
    nationality = None
    nat_match = re.search(r'\b(FRA|FRANÇAISE?|FRENCH)\b', full_text.upper())
    if nat_match:
        nationality = nat_match.group(1)
        logger.info(f"Nationality: '{nationality}'")
    
    # 6. DATE OF BIRTH - 8 digits with spaces
    date_of_birth = None
    birth_date_patterns = [
        r'\b(\d{2})\s*(\d{2})\s*(\d{4})\b',
        r'\b(\d{4})\s*(\d{2})\s*(\d{2})\b',
        r'(\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d\s*\d)',
    ]
    
    birth_candidates = []
    for pattern in birth_date_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            digits_only = re.sub(r'\D', '', match.group())
            
            if len(digits_only) == 8:
                try:
                    # Try DDMMYYYY
                    day = int(digits_only[:2])
                    month = int(digits_only[2:4])
                    year = int(digits_only[4:])
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1950 <= year <= 2010:
                        formatted_date = f"{day:02d}.{month:02d}.{year}"
                        birth_candidates.append((year, formatted_date))
                        continue
                    
                    # Try YYYYMMDD
                    year = int(digits_only[:4])
                    month = int(digits_only[4:6])
                    day = int(digits_only[6:])
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1950 <= year <= 2010:
                        formatted_date = f"{day:02d}.{month:02d}.{year}"
                        birth_candidates.append((year, formatted_date))
                except ValueError:
                    continue
    
    if birth_candidates:
        birth_candidates.sort()
        date_of_birth = birth_candidates[0][1]
        logger.info(f"Date of birth selected: '{date_of_birth}'")
    
    # 7. PLACE OF BIRTH - Bilingual format
    place_of_birth = None
    for i, line in enumerate(lines):
        if 'LIEU DE NAISSANCE' in line.upper() or 'PLACE OF BIRTH' in line.upper():
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 2:
                    place_of_birth = next_line
                    logger.info(f"Place after label: '{place_of_birth}'")
                    break
    
    # 8. EXPIRY DATE - Only on new cards
    expiry_date = None
    for pattern in birth_date_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            digits_only = re.sub(r'\D', '', match.group())
            
            if len(digits_only) == 8:
                try:
                    # Try DDMMYYYY
                    day = int(digits_only[:2])
                    month = int(digits_only[2:4])
                    year = int(digits_only[4:])
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and year >= 2020:
                        expiry_date = f"{day:02d}.{month:02d}.{year}"
                        logger.info(f"Expiry date found: '{expiry_date}'")
                        break
                except ValueError:
                    continue
        
        if expiry_date:
            break
    
    extracted_data = { 
        "Identity number": identity_number if identity_number else "Not found",
        "Surname": surname if surname else "Not found", 
        "First name(s)": first_names if first_names else "Not found", 
        "Sex": sex if sex else "Not found", 
        "Nationality": nationality if nationality else "Not found",
        "Date of birth": date_of_birth if date_of_birth else "Not found", 
        "Place of birth": place_of_birth if place_of_birth else "Not found", 
        "Expiry date": expiry_date if expiry_date else "Not found"
    }
    
    detected_count = sum(1 for v in extracted_data.values() if v != 'Not found')
    logger.info(f"NEW MODEL: Fields detected: {detected_count}/8")
    
    return extracted_data

class DocumentProcessor:
    """Document processor optimized for macOS - with improved extraction"""
    
    def __init__(self):
        self.document_classes = ['ID Card', 'Passport', 'Driver License', 'Other']
        self.has_tesseract = HAS_TESSERACT
        self.has_french = HAS_FRENCH
        logger.info(f"Processor initialized - OCR: {self.has_tesseract}, French: {self.has_french}")
        
    def classify_document(self, image_path):
        """Classification for French identity cards"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}
                
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Analysis of characteristic colors
            blue_score = self._detect_blue_background(image)
            red_score = self._detect_red_elements(image)
            french_id_score = self._detect_french_id_card(image)
            text_score = self._detect_french_text_elements(image)
            
            logger.info(f"Aspect ratio: {aspect_ratio:.2f}, Blue: {blue_score:.3f}, Red: {red_score:.3f}, French ID: {french_id_score:.3f}, Text: {text_score:.3f}")
            
            # Classification logic - priority to French ID indicators
            if (french_id_score > 0.08 or text_score > 0.1 or 
                (blue_score > 0.05 and 1.4 < aspect_ratio < 2.0)):
                doc_type = 'ID Card'
                confidence = min(0.85 + (french_id_score + text_score + blue_score) * 0.3, 0.95)
            elif 1.4 < aspect_ratio < 1.9:
                if red_score > 0.08:  # Driver license
                    doc_type = 'Driver License'
                    confidence = min(0.80 + red_score * 0.15, 0.90)
                else:
                    doc_type = 'ID Card'  # Default for card format
                    confidence = 0.75
            elif aspect_ratio > 2.0 and french_id_score < 0.05 and text_score < 0.05:
                doc_type = 'Passport'
                confidence = 0.70
            else:
                doc_type = 'ID Card'
                confidence = 0.65
                
            all_probabilities = self._generate_probabilities(doc_type, confidence)
            
            logger.info(f"Classification: {doc_type} (confidence: {confidence:.2f})")
                
            return {
                'type': doc_type,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'analysis': {
                    'aspect_ratio': round(aspect_ratio, 2),
                    'blue_score': round(blue_score, 3),
                    'red_score': round(red_score, 3),
                    'french_id_score': round(french_id_score, 3),
                    'text_score': round(text_score, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}

    def _detect_french_text_elements(self, image):
        """Detection of textual elements characteristic of French ID cards"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Contrast enhancement for text detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Detection of horizontal text zones (characteristic of ID cards)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Score based on horizontal line density (structured text)
            line_ratio = np.sum(horizontal_lines > 0) / (horizontal_lines.shape[0] * horizontal_lines.shape[1])
            
            # Bonus for rectangular structure detection (photo frames, etc.)
            edges = cv2.Canny(enhanced, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_shapes = 0
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:
                        rectangular_shapes += 1
            
            # Composite score
            text_score = line_ratio * 0.8 + min(rectangular_shapes * 0.1, 0.3)
            
            return min(text_score, 1.0)
            
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return 0.0
        
    def _detect_french_id_card(self, image):
        """Specific detection for French identity cards"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detection of characteristic blue of French ID cards
            lower_blue1 = np.array([90, 30, 30])
            upper_blue1 = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue1, upper_blue1)
            
            # Detection of red/burgundy French elements (Marianne, etc.)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = red_mask1 + red_mask2
            
            # Score based on presence of characteristic colors
            total_pixels = image.shape[0] * image.shape[1]
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            red_ratio = np.sum(red_mask > 0) / total_pixels
            
            # Composite score for French ID card
            french_score = blue_ratio * 0.7 + red_ratio * 0.3
            
            # Bonus if both colors detected (blue + red = French tricolor)
            if blue_ratio > 0.05 and red_ratio > 0.02:
                french_score *= 1.5
                
            return min(french_score, 1.0)
            
        except Exception as e:
            logger.error(f"French ID detection error: {e}")
            return 0.0
    
    def _detect_blue_background(self, image):
        """Improved blue background detection (French ID card)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Very wide ranges to capture all blues from French ID cards
            lower_blue1 = np.array([80, 20, 30])
            upper_blue1 = np.array([140, 255, 255])
            
            lower_blue2 = np.array([95, 30, 40])
            upper_blue2 = np.array([125, 200, 200])
            
            lower_blue3 = np.array([85, 15, 50])
            upper_blue3 = np.array([130, 100, 150])
            
            mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
            mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
            mask3 = cv2.inRange(hsv, lower_blue3, upper_blue3)
            
            combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
            
            score = np.sum(combined_mask > 0) / (combined_mask.shape[0] * combined_mask.shape[1])
            
            # Bonus if blue found in upper region (header)
            upper_region = combined_mask[:combined_mask.shape[0]//3, :]
            upper_score = np.sum(upper_region > 0) / (upper_region.shape[0] * upper_region.shape[1])
            
            if upper_score > 0.1:
                score *= 1.5
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _detect_red_elements(self, image):
        """Detection of red elements (driver license)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        except:
            return 0.0
    
    def _generate_probabilities(self, predicted_type, confidence):
        """Generate realistic probabilities for all types"""
        probs = {}
        remaining = 1 - confidence
        others = [t for t in self.document_classes if t != predicted_type]
        
        for doc_type in self.document_classes:
            if doc_type == predicted_type:
                probs[doc_type] = confidence
            else:
                if len(others) > 0:
                    probs[doc_type] = remaining / len(others)
                else:
                    probs[doc_type] = 0.0
                
        return probs
    
    def extract_structured_information(self, classification_result, ocr_result):
        """Adaptive information extraction based on ID card model detection"""
        doc_type = classification_result.get('type', 'Other')
        raw_text = ocr_result.get('raw_text', '')
        
        if doc_type == 'ID Card':
            # Step 1: Detect ID card model (old vs new)
            model_type, detection_reason = detect_id_card_model(raw_text)
            
            logger.info(f"ID Card model detected: {model_type} - {detection_reason}")
            
            # Step 2: Use appropriate extraction method
            if model_type == "old":
                extracted_info = extract_info_old_model(raw_text)
                extraction_method = "Old French ID Card extraction (pre-2021)"
                document_subtype = "Old French ID Card (pre-2021)"
            else:
                extracted_info = extract_info_new_model(raw_text)
                extraction_method = "New French ID Card extraction (2021+)"
                document_subtype = "New French ID Card (2021+)"
            
            # Step 3: Add metadata
            extracted_info['document_type'] = 'French National Identity Card'
            extracted_info['document_subtype'] = document_subtype
            extracted_info['card_model'] = model_type
            extracted_info['extraction_method'] = extraction_method
            extracted_info['detection_reason'] = detection_reason
            
            return extracted_info
            
        elif doc_type == 'Passport':
            return {
                'document_type': 'Passport',
                'card_model': 'passport',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Passport parser in development'
            }
        elif doc_type == 'Driver License':
            return {
                'document_type': 'Driver License',
                'card_model': 'license',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Driver license parser in development'
            }
        else:
            return {
                'document_type': 'Unidentified document',
                'card_model': 'unknown',
                'raw_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
            }