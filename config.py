import os
import pytesseract
import logging
from PIL import Image

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # Flask configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    
    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def setup_tesseract():
    """Setup and test Tesseract OCR configuration for macOS/Homebrew"""
    try:
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
        
        has_tesseract = True
        print("✅ Tesseract OCR available and functional")
        
        # Check available languages
        try:
            langs = pytesseract.get_languages()
            print(f"📋 Available languages: {', '.join(langs)}")
            has_french = 'fra' in langs
            if has_french:
                print("✅ French available")
            else:
                print("⚠️ French not available - install with: brew install tesseract-lang")
        except:
            has_french = False
            
    except Exception as e:
        has_tesseract = False
        has_french = False
        print(f"❌ Tesseract not available: {e}")
        print("  To install on macOS:")
        print("   brew install tesseract tesseract-lang")
        print("   pip3 install pytesseract")
    
    return has_tesseract, has_french

# Initialize Tesseract
HAS_TESSERACT, HAS_FRENCH = setup_tesseract()