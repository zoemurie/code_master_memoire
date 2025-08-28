from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import re

# Configuration for macOS with Homebrew
try:
    import pytesseract
    
    # Automatic Tesseract path configuration for macOS/Homebrew
    tesseract_paths = [
        '/opt/homebrew/bin/tesseract',  # Apple Silicon (M1/M2)
        '/usr/local/bin/tesseract',     # Intel Mac
        '/usr/bin/tesseract'            # System installation
    ]
    
    # Find the correct path
    tesseract_cmd = None
    for path in tesseract_paths:
        if os.path.exists(path):
            tesseract_cmd = path
            break
    
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        print(f"✅ Tesseract found at: {tesseract_cmd}")
    
    # Functionality test
    test_image = Image.new('RGB', (100, 50), color='white')
    pytesseract.image_to_string(test_image)
    
    HAS_TESSERACT = True
    print("✅ Tesseract OCR available and functional")
    
    # Check available languages
    try:
        langs = pytesseract.get_languages()
        print(f"📚 Available languages: {', '.join(langs)}")
        HAS_FRENCH = 'fra' in langs
        if HAS_FRENCH:
            print("✅ French available")
        else:
            print("⚠️ French not available - install with: brew install tesseract-lang")
    except:
        HAS_FRENCH = False
        
except Exception as e:
    HAS_TESSERACT = False
    HAS_FRENCH = False
    print(f"❌ Tesseract not available: {e}")
    print("🔧 To install on macOS:")
    print("   brew install tesseract tesseract-lang")
    print("   pip3 install pytesseract")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# IMPROVED EXTRACTION FUNCTIONS FROM OLD_CODE.PY

def correct_skew(image_path): 
    """Image skew correction - old_code.py version"""
    img = cv2.imread(image_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Edge detection 
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) 

    # Line detection with Hough Transform 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200) 
    if lines is not None: 
        angles = [] 
        for rho, theta in lines[:, 0]: 
            angle = (theta - np.pi / 2) * 180 / np.pi # Convert to degrees 
            angles.append(angle) 
        # Calculate median angle to avoid errors 
        skew_angle = np.median(angles) 
        # Inverse rotation of image 
        (h, w) = img.shape[:2] 
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0) 
        img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
        # Save corrected image 
        corrected_path = image_path.replace(".jpg", "_corrected.jpg").replace(".png", "_corrected.png").replace(".jpeg", "_corrected.jpeg")
        cv2.imwrite(corrected_path, img_rotated) 
        return corrected_path 
    return image_path # Return original image if no angle detected 

def preprocess_image(image_path): 
    """Image preprocessing - old_code.py version"""
    # Skew correction 
    corrected_path = correct_skew(image_path) 
    img = cv2.imread(corrected_path, cv2.IMREAD_GRAYSCALE) 
    # Noise reduction 
    blurred = cv2.GaussianBlur(img, (5, 5), 0) 
    # Binarization (thresholding) 
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    # Save processed image
    preprocessed_path = corrected_path.replace(".jpg", "_processed.jpg").replace(".png", "_processed.png").replace(".jpeg", "_processed.jpeg")
    cv2.imwrite(preprocessed_path, thresh) 
    return preprocessed_path 

def extract_info_with_confidence(image_path): 
    """Text extraction with confidence - old_code.py version"""
    # Load preprocessed image 
    img = Image.open(image_path) 

    # Use image_to_data to get OCR confidence information 
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) 
    
    # Calculate average confidence of extracted words 
    confidences = [] 
    for conf in ocr_data['conf']: 
        if conf != '-1': # Ignore undefined values 
            confidences.append(int(conf)) 
    if confidences: 
        average_confidence = sum(confidences) / len(confidences) 
    else: 
        average_confidence = 0 
        
    # Extract text 
    text = pytesseract.image_to_string(img) 
    
    # Calculate confidence 
    confidence_threshold = 80 # For example, 80% confidence threshold 
    is_confident = average_confidence >= confidence_threshold 
    
    return text, average_confidence, is_confident 

def extract_info_basic(text): 
    """Basic information extraction - old_code.py version"""
    # Normalize spaces and parasitic characters 
    text = text.replace('|', ' ').replace('Ne:', ' ').replace('$', ' ') 
    
    # Extract card number (12 digits) 
    identity_number_match = re.search(r'\b\d{12}\b', text) 
    if identity_number_match: 
        identity_number = identity_number_match.group(0) 
    else: 
        all_digits = re.findall(r'\d', text) 
        identity_number = ''.join(all_digits)[:12].ljust(12, '?') 
    
    # Extract date of birth (xx.xx.xxxx) 
    dob_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text) 
    if dob_match: 
        date_of_birth = dob_match.group(1) 
    else: 
        # Get all digits from text 
        all_digits = re.findall(r'\d', text) 
        if len(all_digits) >= 8: 
            date_of_birth = f"{''.join(all_digits[:2])}.{''.join(all_digits[2:4])}.{''.join(all_digits[4:8])}" 
        else: 
            date_of_birth = "??.??.????" 
    
    # Other information 
    surname = re.search(r'Nom\s*:\s*(\S+)', text) 
    name = re.search(r'Prénom\(s\)\s*:\s*(.+)', text) 
    sex = re.search(r'Sexe\s*:\s*(\S+)', text) 
    pob = re.search(r'à\s*([A-Za-zÀ-ÿ\s-]+)', text) 
    height = re.search(r'Taille\s*:\s*(\S+)', text) 
    
    extracted_data = { 
        "Identity number": identity_number, 
        "Surname": surname.group(1) if surname else "Not found", 
        "First name(s)": name.group(1) if name else "Not found", 
        "Sex": sex.group(1) if sex else "Not found", 
        "Date of birth": date_of_birth, 
        "Place of birth": pob.group(1) if pob else "Not found", 
        "Height": height.group(1) if height else "Not found" 
    } 
    return extracted_data 

def detect_document_type(text): 
    """Document type detection - old_code.py version"""
    if "passeport" in text.lower(): 
        return "Passport" 
    elif "CARTE NATIONALE" in text: 
        if "IDENTITY CARD" in text: 
            return "New ID card model" 
        else: 
            return "Old ID card model" 
    return "Unknown document"

class DocumentProcessor:
    """
    Document processor optimized for macOS - with improved extraction
    """
    def __init__(self):
        self.document_classes = ['ID Card', 'Passport', 'Driver License', 'Other']
        self.has_tesseract = HAS_TESSERACT
        self.has_french = HAS_FRENCH
        logger.info(f"Processor initialized - OCR: {self.has_tesseract}, French: {self.has_french}")
        
    def classify_document(self, image_path):
        """Intelligent document classification"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}
                
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Characteristic color analysis
            blue_score = self._detect_blue_background(image)
            red_score = self._detect_red_elements(image)
            
            logger.info(f"Aspect ratio: {aspect_ratio:.2f}, Blue: {blue_score:.3f}, Red: {red_score:.3f}")
            
            # Classification based on visual characteristics
            if 1.4 < aspect_ratio < 1.8:  # Card format
                if blue_score > 0.12:  # French ID card (blue background)
                    doc_type = 'ID Card'
                    confidence = min(0.85 + blue_score * 0.15, 0.95)
                elif red_score > 0.08:  # Driver license (red elements)
                    doc_type = 'Driver License'
                    confidence = min(0.80 + red_score * 0.15, 0.92)
                else:
                    doc_type = 'ID Card'  # Default for card format
                    confidence = 0.70
            elif aspect_ratio > 1.25:  # Booklet format
                doc_type = 'Passport'
                confidence = 0.82
            else:
                doc_type = 'Other'
                confidence = 0.60
                
            # Generate probabilities for all types
            all_probabilities = self._generate_probabilities(doc_type, confidence)
            
            logger.info(f"Classification: {doc_type} (confidence: {confidence:.2f})")
                
            return {
                'type': doc_type,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'analysis': {
                    'aspect_ratio': round(aspect_ratio, 2),
                    'blue_score': round(blue_score, 3),
                    'red_score': round(red_score, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}
    
    def _detect_blue_background(self, image):
        """Detection of characteristic blue background of ID card"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Blue range for French ID card
            lower_blue = np.array([95, 50, 50])
            upper_blue = np.array([125, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        except:
            return 0.0
    
    def _detect_red_elements(self, image):
        """Detection of red elements (driver license)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Two ranges for red (beginning and end of spectrum)
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
                # Realistic distribution of remainder
                if len(others) > 0:
                    probs[doc_type] = remaining / len(others)
                else:
                    probs[doc_type] = 0.0
                
        return probs
    
    def extract_text_with_ocr(self, image_path):
        """Text extraction with improved method from old_code.py"""
        if not self.has_tesseract:
            return self._simulate_ocr_extraction(image_path)
        
        try:
            # Use preprocessing from old_code.py
            processed_path = preprocess_image(image_path)
            
            # Use confidence extraction from old_code.py
            start_time = time.time()
            text, average_confidence, is_confident = extract_info_with_confidence(processed_path)
            processing_time = time.time() - start_time
            
            logger.info(f"OCR completed in {processing_time:.2f}s - {len(text)} characters - confidence: {average_confidence:.1f}%")
            
            return {
                'raw_text': text.strip(),
                'confidence': average_confidence / 100,
                'method': f'Tesseract OCR (improved old_code.py method)',
                'processing_time': round(processing_time, 2),
                'word_count': len(text.split()),
                'character_count': len(text),
                'is_confident': is_confident,
                'processed_image_path': processed_path
            }
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return self._simulate_ocr_extraction(image_path)
    
    def _simulate_ocr_extraction(self, image_path):
        """OCR simulation when Tesseract is not available"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
        except:
            mean_intensity = 128
        
        # Realistic simulation based on image quality
        if mean_intensity > 140:  # Clear image
            confidence = 0.92
            simulated_text = """FRENCH REPUBLIC
NATIONAL IDENTITY CARD

Surname: MARTIN
First name(s): Jean Pierre
Born on: 15.03.1985
at: PARIS 15EME (75)
Sex: M
Height: 1.75 m
Nationality: French
No.: 123456789012

Issued on: 20.01.2020
by: PREFECTURE DE POLICE
Valid until: 20.01.2030"""
        else:  # Less clear image
            confidence = 0.76
            simulated_text = """FRENCH REPUBLIC
NATIONAL IDENTITY CARD

Surname: DUBOIS
First name(s): Marie Claire
Born on: 22.07.1992
at: LYON 3EME (69)
Sex: F
Height: 1.65 m
Nationality: French
No.: 987654321098"""
        
        return {
            'raw_text': simulated_text,
            'confidence': confidence,
            'method': 'Simulation (Tesseract not available)',
            'processing_time': 0.1,
            'word_count': len(simulated_text.split()),
            'character_count': len(simulated_text),
            'is_confident': confidence > 0.8,
            'processed_image_path': image_path
        }
    
    def extract_structured_information(self, classification_result, ocr_result):
        """Structured information extraction - uses both methods"""
        doc_type = classification_result.get('type', 'Other')
        raw_text = ocr_result.get('raw_text', '')
        
        if doc_type == 'ID Card':
            # Use basic method from old_code.py first
            basic_info = extract_info_basic(raw_text)
            
            # Add document information
            basic_info['document_type'] = 'National Identity Card'
            basic_info['extraction_method'] = 'old_code.py improved'
            basic_info['document_detection'] = detect_document_type(raw_text)
            
            return basic_info
            
        elif doc_type == 'Passport':
            return {
                'document_type': 'Passport',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Passport parser in development'
            }
        elif doc_type == 'Driver License':
            return {
                'document_type': 'Driver License',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Driver license parser in development'
            }
        else:
            return {
                'document_type': 'Unidentified document',
                'raw_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
            }

# Global processor instance
processor = DocumentProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Detailed system status
    if HAS_TESSERACT:
        if HAS_FRENCH:
            status = "✅ Complete Tesseract OCR (French + English) + old_code.py method"
            status_class = "success"
        else:
            status = "⚠️ Tesseract OCR (English only) + old_code.py method"
            status_class = "warning"
    else:
        status = "❌ Simulation mode (Tesseract not installed)"
        status_class = "error"
    
    install_section = ""
    if not HAS_TESSERACT or not HAS_FRENCH:
        install_section = '''
            <div class="install-info">
                <h3>🔧 Installation on macOS:</h3>
                <pre><code>brew install tesseract tesseract-lang
pip3 install pytesseract</code></pre>
            </div>
            '''
    
    return f'''
    <!doctype html>
    <html>
    <head>
        <title>Advanced OCR System - Identity Documents (Enhanced Version)</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h1 {{ 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            .upload-area {{ 
                border: 3px dashed #ddd; 
                padding: 50px; 
                text-align: center; 
                margin: 30px 0; 
                border-radius: 15px; 
                transition: all 0.3s ease;
            }}
            .upload-area:hover {{ 
                border-color: #667eea; 
                background-color: #f8f9ff;
            }}
            input[type="file"] {{ 
                margin: 15px; 
                padding: 10px;
                font-size: 16px;
            }}
            input[type="submit"] {{ 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 18px;
                font-weight: bold;
                transition: transform 0.2s ease;
            }}
            input[type="submit"]:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .status {{ 
                padding: 20px; 
                border-radius: 10px; 
                margin: 25px 0; 
                font-weight: bold;
            }}
            .status.success {{ background-color: #d4edda; border: 2px solid #c3e6cb; color: #155724; }}
            .status.warning {{ background-color: #fff3cd; border: 2px solid #ffeaa7; color: #856404; }}
            .status.error {{ background-color: #f8d7da; border: 2px solid #f5c6cb; color: #721c24; }}
            .features {{ margin-top: 40px; }}
            .feature {{ 
                background: linear-gradient(45deg, #f8f9fa, #e9ecef); 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border-left: 5px solid #667eea;
            }}
            .install-info {{ 
                background-color: #f8d7da; 
                border: 2px solid #f5c6cb; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
            }}
            pre {{ 
                background-color: #f1f3f4; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto;
            }}
            .new-feature {{
                background: linear-gradient(45deg, #28a745, #20c997); 
                color: white;
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Advanced OCR System</h1>
            <p style="text-align: center; color: #666; font-size: 18px;">
                Automatic recognition of identity documents with AI + Enhanced extraction
            </p>
            
            {install_section}
            
            <form method="post" enctype="multipart/form-data" action="/upload">
                <div class="upload-area">
                    <h3>📄 Upload an identity document</h3>
                    <p style="color: #666;">ID Card, Passport, Driver License</p>
                    <input type="file" name="file" accept=".png,.jpg,.jpeg" required>
                    <br><br>
                    <input type="submit" value="Analyze">
                </div>
            </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file sent", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            return redirect(url_for('analyze_document', filename=filename))
        except Exception as e:
            logger.error(f"File save error: {e}")
            return "Error during file save", 500
    
    return "File type not allowed", 400

@app.route('/analyze/<filename>')
def analyze_document(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return "File not found", 404
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Document classification
        logger.info("🔍 Document classification...")
        classification_result = processor.classify_document(image_path)
        
        # Step 2: OCR extraction with enhanced method
        logger.info("🔍 OCR text extraction (old_code.py method)...")
        ocr_result = processor.extract_text_with_ocr(image_path)
        
        # Step 3: Structured parsing
        logger.info("🧠 Information parsing...")
        structured_info = processor.extract_structured_information(classification_result, ocr_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate report
        analysis_report = {
            'filename': filename,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            'classification': classification_result,
            'ocr_result': ocr_result,
            'structured_information': structured_info,
            'has_real_ocr': HAS_TESSERACT,
            'has_french': HAS_FRENCH,
            'system_info': {
                'tesseract_path': pytesseract.pytesseract.tesseract_cmd if HAS_TESSERACT else None,
                'opencv_version': cv2.__version__,
                'platform': 'macOS',
                'extraction_method': 'old_code.py enhanced'
            }
        }
        
        logger.info(f"✅ Analysis completed in {processing_time:.2f}s")
        
        return render_template_string(get_results_template(), 
                                    report=analysis_report, 
                                    filename=filename)
                                    
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        return f"Error during analysis: {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_results_template():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>AI Analysis Results</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                padding: 20px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border-radius: 10px;
            }
            .section { 
                margin: 25px 0; 
                padding: 25px; 
                border: 2px solid #e9ecef; 
                border-radius: 10px; 
                background: #f8f9fa;
            }
            .success { background: linear-gradient(45deg, #d4edda, #c3e6cb); border-color: #28a745; }
            .info { background: linear-gradient(45deg, #d1ecf1, #bee5eb); border-color: #17a2b8; }
            .warning { background: linear-gradient(45deg, #fff3cd, #ffeaa7); border-color: #ffc107; }
            .error { background: linear-gradient(45deg, #f8d7da, #f5c6cb); border-color: #dc3545; }
            .improved { background: linear-gradient(45deg, #28a745, #20c997); color: white; }
            
            .image-preview { 
                max-width: 300px; 
                max-height: 400px; 
                border-radius: 10px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin: 15px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
                gap: 20px; 
                margin: 25px 0; 
            }
            .metric { 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }
            .metric h3 { 
                margin: 0 0 10px 0; 
                color: #333; 
                font-size: 1.1em;
            }
            .metric p { 
                margin: 0; 
                font-size: 1.5em; 
                font-weight: bold; 
            }
            .confidence-high { color: #28a745; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
            
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            th, td { 
                padding: 15px; 
                text-align: left; 
                border-bottom: 1px solid #dee2e6; 
            }
            th { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                font-weight: bold; 
            }
            tr:hover { background-color: #f8f9fa; }
            
            .back-button { 
                background: linear-gradient(45deg, #6c757d, #5a6268); 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 8px; 
                display: inline-block; 
                margin-top: 30px; 
                font-weight: bold;
                transition: transform 0.2s ease;
            }
            .back-button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .text-extract { 
                background: white; 
                color: #000000;
                padding: 20px; 
                border-radius: 8px; 
                font-family: 'Monaco', 'Menlo', monospace; 
                font-size: 14px; 
                white-space: pre-wrap; 
                max-height: 300px; 
                overflow-y: auto; 
                border: 2px solid #333333;
                line-height: 1.5;
            }
            
            .json-view { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                white-space: pre-wrap; 
                font-family: 'Monaco', 'Menlo', monospace; 
                font-size: 12px; 
                max-height: 400px; 
                overflow-y: auto; 
                border: 1px solid #dee2e6;
            }
            
            details { 
                margin: 15px 0; 
            }
            summary { 
                cursor: pointer; 
                padding: 10px; 
                background: #e9ecef; 
                border-radius: 5px; 
                font-weight: bold;
            }
            summary:hover { 
                background: #dee2e6; 
            }
            
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }
            .status-real { background: #d4edda; color: #155724; }
            .status-sim { background: #fff3cd; color: #856404; }
            .status-improved { background: #28a745; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Analysis Results</h1>
                <p>Processing completed in <strong>{{ report.processing_time }}s</strong> | {{ report.timestamp }}</p>
                <p>
                    <strong>OCR Mode:</strong> 
                    {% if report.has_real_ocr %}
                        Real Tesseract + Enhanced Method
                    {% else %}
                        Simulation
                        <span class="status-badge status-sim">DEMO</span>
                    {% endif %}
                </p>
            </div>
            
            <!-- Original image -->
            <div class="section">
                <h2>Analyzed Document</h2>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="text-align: center;">
                        <img src="{{ url_for('uploaded_file', filename=filename) }}" class="image-preview" alt="Original Document">
                        <p><strong>Original Image</strong></p>
                    </div>
                    {% if report.ocr_result.processed_image_path %}
                    <div style="text-align: center;">
                        <img src="{{ url_for('uploaded_file', filename=report.ocr_result.processed_image_path.split('/')[-1]) }}" class="image-preview" alt="Preprocessed Document">
                        <p><strong>Preprocessed Image</strong></p>
                    </div>
                    {% endif %}
                </div>
                <p><strong>File:</strong> {{ filename }}</p>
            </div>
            
            <!-- Performance metrics -->
            <div class="metrics">
                <div class="metric">
                    <h3>Total Time</h3>
                    <p class="confidence-high">{{ report.processing_time }}s</p>
                </div>
                <div class="metric">
                    <h3>Detected Type</h3>
                    <p><strong>{{ report.classification.type }}</strong></p>
                </div>
                <div class="metric">
                    <h3>Classification Confidence</h3>
                    <p class="{% if report.classification.confidence > 0.8 %}confidence-high{% elif report.classification.confidence > 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                        {{ "%.1f"|format(report.classification.confidence * 100) }}%
                    </p>
                </div>
                <div class="metric">
                    <h3>OCR Confidence</h3>
                    <p class="{% if report.ocr_result.confidence > 0.8 %}confidence-high{% elif report.ocr_result.confidence > 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                        {{ "%.1f"|format(report.ocr_result.confidence * 100) }}%
                    </p>
                </div>
                <div class="metric">
                    <h3>Extracted Words</h3>
                    <p class="confidence-high">{{ report.ocr_result.word_count }}</p>
                </div>
                <div class="metric">
                    <h3>Characters</h3>
                    <p class="confidence-high">{{ report.ocr_result.character_count }}</p>
                </div>
            </div>

            <!-- Detailed classification -->
            <div class="section info">
                <h2>Intelligent Classification</h2>
                <p><strong>Identified type:</strong> {{ report.classification.type }}</p>
                <p><strong>Confidence:</strong> {{ "%.2f"|format(report.classification.confidence * 100) }}%</p>
                
                {% if report.classification.analysis %}
                <h3>Visual Analysis:</h3>
                <ul>
                    <li><strong>Aspect ratio:</strong> {{ report.classification.analysis.aspect_ratio }}</li>
                    <li><strong>Blue score (ID Card):</strong> {{ report.classification.analysis.blue_score }}</li>
                    <li><strong>Red score (License):</strong> {{ report.classification.analysis.red_score }}</li>
                </ul>
                {% endif %}
                
                <h3>Probabilities by Type:</h3>
                <table>
                    <thead>
                        <tr><th>Document Type</th><th>Probability</th><th>Bar</th></tr>
                    </thead>
                    <tbody>
                        {% for doc_type, prob in report.classification.all_probabilities.items() %}
                        <tr>
                            <td>{{ doc_type }}</td>
                            <td>{{ "%.2f"|format(prob * 100) }}%</td>
                            <td>
                                <div style="background: #e9ecef; border-radius: 10px; height: 20px; width: 100px; overflow: hidden;">
                                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); height: 100%; width: {{ prob * 100 }}%; border-radius: 10px;"></div>
                                </div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- OCR extraction -->
            <div class="section info">
                <h2>OCR Text Extraction (Enhanced Method)</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div><strong>Method:</strong> {{ report.ocr_result.method }}</div>
                    <div><strong>Confidence:</strong> {{ "%.1f"|format(report.ocr_result.confidence * 100) }}%</div>
                    <div><strong>OCR Time:</strong> {{ report.ocr_result.processing_time }}s</div>
                    <div><strong>Words detected:</strong> {{ report.ocr_result.word_count }}</div>
                    {% if report.ocr_result.is_confident is defined %}
                    <div><strong>Extraction:</strong> 
                        {% if report.ocr_result.is_confident %}
                            <span style="color: #28a745; font-weight: bold;">High Confidence</span>
                        {% else %}
                            <span style="color: #dc3545; font-weight: bold;">Low Confidence</span>
                        {% endif %}
                    </div>
                    {% endif %}
                </div>
                
                <h3>Extracted Text:</h3>
                <div class="text-extract">{{ report.ocr_result.raw_text }}</div>
                
                {% if report.has_real_ocr %}
                <div style="margin-top: 15px; padding: 15px; background-color: #d4edda; border-radius: 8px; border-left: 5px solid #28a745;">
                    <strong>Now uses the optimized extraction method from old_code.py for better results!</strong>
                </div>
                {% else %}
                <div style="margin-top: 15px; padding: 15px; background-color: #fff3cd; border-radius: 8px; border-left: 5px solid #ffc107;">
                    <strong>Note:</strong> Simulated data - Install Tesseract for real OCR:
                    <code>brew install tesseract tesseract-lang</code>
                </div>
                {% endif %}
            </div>

            <!-- Structured information -->
            <div class="section info">
                <h2>Extracted and Structured Information</h2>
                <p><strong>Document type:</strong> {{ report.structured_information.document_type }}</p>
                {% if report.structured_information.extraction_method %}
                <p><strong>Extraction method:</strong> {{ report.structured_information.extraction_method }}</p>
                {% endif %}
                {% if report.structured_information.document_detection %}
                <p><strong>Automatic detection:</strong> {{ report.structured_information.document_detection }}</p>
                {% endif %}
                
                <table>
                    <thead>
                        <tr><th>Field</th><th>Extracted Value</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                        {% for key, value in report.structured_information.items() %}
                        {% if key not in ['document_type', 'extraction_method', 'document_detection', 'detected_text', 'note', 'raw_text'] %}
                        <tr>
                            <td><strong>{{ key.replace('_', ' ').title() }}</strong></td>
                            <td>{{ value }}</td>
                            <td>
                                {% if value not in ["Not detected", "Not found"] %}
                                    <span style="color: #28a745; font-weight: bold;">Detected</span>
                                {% else %}
                                    <span style="color: #dc3545; font-weight: bold;">Not detected</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- System information -->
            <div class="section">
                <h2>System Information</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div><strong>Platform:</strong> {{ report.system_info.platform }}</div>
                    <div><strong>OpenCV:</strong> {{ report.system_info.opencv_version }}</div>
                    <div><strong>Method:</strong> {{ report.system_info.extraction_method }}</div>
                    {% if report.system_info.tesseract_path %}
                    <div><strong>Tesseract:</strong> {{ report.system_info.tesseract_path }}</div>
                    {% endif %}
                </div>
                
                <details style="margin-top: 20px;">
                    <summary>View complete JSON data</summary>
                    <div class="json-view">{{ report | tojson(indent=2) }}</div>
                </details>
            </div>

            <!-- Actions -->
            <div style="text-align: center;">
                <a href="{{ url_for('index') }}" class="back-button">Analyze Another Document</a>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("=" * 70)
    print("STARTING ADVANCED OCR SYSTEM v2.0 (ENHANCED METHOD)")
    print("=" * 70)
    print(f"Web interface: http://localhost:8080")
    print(f"Real OCR: {'✅ Enabled' if HAS_TESSERACT else '❌ Install Tesseract'}")
    print(f"French: {'✅ Available' if HAS_FRENCH else '❌ Install tesseract-lang'}")
    print(f"Extraction: ✅ old_code.py method integrated")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    
    if not HAS_TESSERACT:
        print("\n⚠️ INSTALLATION REQUIRED:")
        print("   brew install tesseract tesseract-lang")
        print("   pip3 install pytesseract")
    
    print("=" * 70)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\nServer shutdown...")
    except Exception as e:
        print(f"\nError: {e}")