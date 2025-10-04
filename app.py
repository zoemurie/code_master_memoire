from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import pytesseract

from config import Config, HAS_TESSERACT, HAS_FRENCH
from ocr_processor import extract_text_with_ocr
from document_analyzer import DocumentProcessor

# Logging configuration
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Apply configuration
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize document processor
processor = DocumentProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page with upload form"""
    # System status for template
    if HAS_TESSERACT:
        if HAS_FRENCH:
            status = "Complete Tesseract OCR (French + English) + old_code.py method"
            status_class = "success"
        else:
            status = "Tesseract OCR (English only) + old_code.py method"
            status_class = "warning"
    else:
        status = "Simulation mode (Tesseract not installed)"
        status_class = "error"
    
    show_install = not HAS_TESSERACT or not HAS_FRENCH
    
    return render_template('index.html', 
                         status=status, 
                         status_class=status_class, 
                         show_install=show_install)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
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
    """Analyze uploaded document"""
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return "File not found", 404
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Document classification
        logger.info("Document classification...")
        classification_result = processor.classify_document(image_path)
        
        # Step 2: OCR extraction with enhanced method
        logger.info("OCR text extraction (old_code.py method)...")
        ocr_result = extract_text_with_ocr(image_path)
        
        # Step 3: Structured parsing
        logger.info("Information parsing...")
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
                'tesseract_path': getattr(__import__('pytesseract').pytesseract, 'tesseract_cmd', None) if HAS_TESSERACT else None,
                'opencv_version': __import__('cv2').__version__,
                'platform': 'macOS',
                'extraction_method': 'old_code.py enhanced'
            }
        }
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        return render_template('results.html', 
                             report=analysis_report, 
                             filename=filename)
                                    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return f"Error during analysis: {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("=" * 70)
    print("STARTING ADVANCED OCR SYSTEM v2.0 (ENHANCED METHOD)")
    print("=" * 70)
    print(f"Web interface: http://localhost:8080")
    if HAS_TESSERACT:
        print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    print(f"Real OCR: {'✅ Enabled' if HAS_TESSERACT else '❌ Install Tesseract'}")
    print(f"French: {'✅ Available' if HAS_FRENCH else '❌ Install tesseract-lang'}")
    print(f"Extraction: ✅ old_code.py method integrated")
    print(f"Upload folder: {os.path.abspath(Config.UPLOAD_FOLDER)}")
    
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