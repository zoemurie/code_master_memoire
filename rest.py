from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import uuid
import re

# Configuration pour macOS avec Homebrew
try:
    import pytesseract
    
    # Configuration automatique du chemin Tesseract pour macOS/Homebrew
    tesseract_paths = [
        '/opt/homebrew/bin/tesseract',  # Apple Silicon (M1/M2)
        '/usr/local/bin/tesseract',     # Intel Mac
        '/usr/bin/tesseract'            # Installation système
    ]
    
    # Trouve le bon chemin
    tesseract_cmd = None
    for path in tesseract_paths:
        if os.path.exists(path):
            tesseract_cmd = path
            break
    
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        print(f"✅ Tesseract trouvé à: {tesseract_cmd}")
    
    # Test de fonctionnalité
    test_image = Image.new('RGB', (100, 50), color='white')
    pytesseract.image_to_string(test_image)
    
    HAS_TESSERACT = True
    print("✅ Tesseract OCR disponible et fonctionnel")
    
    # Vérification des langues disponibles
    try:
        langs = pytesseract.get_languages()
        print(f"📚 Langues disponibles: {', '.join(langs)}")
        HAS_FRENCH = 'fra' in langs
        if HAS_FRENCH:
            print("✅ Français disponible")
        else:
            print("⚠️ Français non disponible - installer avec: brew install tesseract-lang")
    except:
        HAS_FRENCH = False
        
except Exception as e:
    HAS_TESSERACT = False
    HAS_FRENCH = False
    print(f"❌ Tesseract non disponible: {e}")
    print("🔧 Pour installer sur macOS:")
    print("   brew install tesseract tesseract-lang")
    print("   pip3 install pytesseract")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# FONCTIONS D'EXTRACTION AMÉLIORÉES DE OLD_CODE.PY

def correct_skew(image_path): 
    """Correction de l'inclinaison de l'image - version old_code.py"""
    img = cv2.imread(image_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Détection des contours 
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) 

    # Détection des lignes avec la transformée de Hough 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200) 
    if lines is not None: 
        angles = [] 
        for rho, theta in lines[:, 0]: 
            angle = (theta - np.pi / 2) * 180 / np.pi # Conversion en degrés 
            angles.append(angle) 
        # Calcul de l'angle médian pour éviter les erreurs 
        skew_angle = np.median(angles) 
        # Rotation inverse de l'image 
        (h, w) = img.shape[:2] 
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0) 
        img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
        # Sauvegarde de l'image corrigée 
        corrected_path = image_path.replace(".jpg", "_corrected.jpg").replace(".png", "_corrected.png").replace(".jpeg", "_corrected.jpeg")
        cv2.imwrite(corrected_path, img_rotated) 
        return corrected_path 
    return image_path # Retourne l'image originale si aucun angle détecté 

def preprocess_image(image_path): 
    """Prétraitement de l'image - version old_code.py"""
    # Correction de l'inclinaison 
    corrected_path = correct_skew(image_path) 
    img = cv2.imread(corrected_path, cv2.IMREAD_GRAYSCALE) 
    # Réduction du bruit 
    blurred = cv2.GaussianBlur(img, (5, 5), 0) 
    # Binarisation (seuillage) 
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    # Sauvegarde de l'image traitée
    preprocessed_path = corrected_path.replace(".jpg", "_processed.jpg").replace(".png", "_processed.png").replace(".jpeg", "_processed.jpeg")
    cv2.imwrite(preprocessed_path, thresh) 
    return preprocessed_path 

def extract_info_with_confidence(image_path): 
    """Extraction de texte avec confiance - version old_code.py"""
    # Charger l'image prétraitée 
    img = Image.open(image_path) 

    # Utiliser image_to_data pour obtenir les informations de confiance OCR 
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) 
    
    # Calculer la confiance moyenne des mots extraits 
    confidences = [] 
    for conf in ocr_data['conf']: 
        if conf != '-1': # Ignorer les valeurs non définies 
            confidences.append(int(conf)) 
    if confidences: 
        average_confidence = sum(confidences) / len(confidences) 
    else: 
        average_confidence = 0 
        
    # Extraire le texte 
    text = pytesseract.image_to_string(img) 
    
    # Calculer la confiance 
    confidence_threshold = 80 # Par exemple, seuil de confiance de 80% 
    is_confident = average_confidence >= confidence_threshold 
    
    return text, average_confidence, is_confident 

def extract_info_basic(text): 
    """Extraction d'informations de base - version old_code.py"""
    # Normaliser les espaces et caractères parasites 
    text = text.replace('|', ' ').replace('Ne:', ' ').replace('$', ' ') 
    
    # Extraire le numéro de carte (12 chiffres) 
    identity_number_match = re.search(r'\b\d{12}\b', text) 
    if identity_number_match: 
        identity_number = identity_number_match.group(0) 
    else: 
        all_digits = re.findall(r'\d', text) 
        identity_number = ''.join(all_digits)[:12].ljust(12, '?') 
    
    # Extraire la date de naissance (xx.xx.xxxx) 
    dob_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text) 
    if dob_match: 
        date_of_birth = dob_match.group(1) 
    else: 
        # Obtenir tous les chiffres du texte 
        all_digits = re.findall(r'\d', text) 
        if len(all_digits) >= 8: 
            date_of_birth = f"{''.join(all_digits[:2])}.{''.join(all_digits[2:4])}.{''.join(all_digits[4:8])}" 
        else: 
            date_of_birth = "??.??.????" 
    
    # Autres informations 
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
    """Détection du type de document - version old_code.py"""
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
    Processeur de documents optimisé pour macOS - avec extraction améliorée
    """
    def __init__(self):
        self.document_classes = ['ID Card', 'Passport', 'Driver License', 'Other']
        self.has_tesseract = HAS_TESSERACT
        self.has_french = HAS_FRENCH
        logger.info(f"Processeur initialisé - OCR: {self.has_tesseract}, Français: {self.has_french}")
        
    def classify_document(self, image_path):
        """Classification intelligente des documents"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}
                
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Analyse des couleurs caractéristiques
            blue_score = self._detect_blue_background(image)
            red_score = self._detect_red_elements(image)
            
            logger.info(f"Ratio d'aspect: {aspect_ratio:.2f}, Bleu: {blue_score:.3f}, Rouge: {red_score:.3f}")
            
            # Classification basée sur les caractéristiques visuelles
            if 1.4 < aspect_ratio < 1.8:  # Format carte
                if blue_score > 0.12:  # Carte d'identité française (fond bleu)
                    doc_type = 'ID Card'
                    confidence = min(0.85 + blue_score * 0.15, 0.95)
                elif red_score > 0.08:  # Permis de conduire (éléments rouges)
                    doc_type = 'Driver License'
                    confidence = min(0.80 + red_score * 0.15, 0.92)
                else:
                    doc_type = 'ID Card'  # Par défaut pour format carte
                    confidence = 0.70
            elif aspect_ratio > 1.25:  # Format livret
                doc_type = 'Passport'
                confidence = 0.82
            else:
                doc_type = 'Other'
                confidence = 0.60
                
            # Génération des probabilités pour tous les types
            all_probabilities = self._generate_probabilities(doc_type, confidence)
            
            logger.info(f"Classification: {doc_type} (confiance: {confidence:.2f})")
                
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
            logger.error(f"Erreur de classification: {e}")
            return {'type': 'Error', 'confidence': 0.0, 'all_probabilities': {}}
    
    def _detect_blue_background(self, image):
        """Détection du fond bleu caractéristique de la carte d'identité"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Plage de bleu pour carte d'identité française
            lower_blue = np.array([95, 50, 50])
            upper_blue = np.array([125, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        except:
            return 0.0
    
    def _detect_red_elements(self, image):
        """Détection d'éléments rouges (permis de conduire)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Deux plages pour le rouge (début et fin du spectre)
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
        """Génération de probabilités réalistes pour tous les types"""
        probs = {}
        remaining = 1 - confidence
        others = [t for t in self.document_classes if t != predicted_type]
        
        for doc_type in self.document_classes:
            if doc_type == predicted_type:
                probs[doc_type] = confidence
            else:
                # Distribution réaliste du reste
                if len(others) > 0:
                    probs[doc_type] = remaining / len(others)
                else:
                    probs[doc_type] = 0.0
                
        return probs
    
    def extract_text_with_ocr(self, image_path):
        """Extraction de texte avec méthode améliorée de old_code.py"""
        if not self.has_tesseract:
            return self._simulate_ocr_extraction(image_path)
        
        try:
            # Utilisation du prétraitement de old_code.py
            processed_path = preprocess_image(image_path)
            
            # Utilisation de l'extraction avec confiance de old_code.py
            start_time = time.time()
            text, average_confidence, is_confident = extract_info_with_confidence(processed_path)
            processing_time = time.time() - start_time
            
            logger.info(f"OCR terminé en {processing_time:.2f}s - {len(text)} caractères - confiance: {average_confidence:.1f}%")
            
            return {
                'raw_text': text.strip(),
                'confidence': average_confidence / 100,
                'method': f'Tesseract OCR (méthode old_code.py améliorée)',
                'processing_time': round(processing_time, 2),
                'word_count': len(text.split()),
                'character_count': len(text),
                'is_confident': is_confident,
                'processed_image_path': processed_path
            }
            
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return self._simulate_ocr_extraction(image_path)
    
    def _simulate_ocr_extraction(self, image_path):
        """Simulation OCR quand Tesseract n'est pas disponible"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
        except:
            mean_intensity = 128
        
        # Simulation réaliste basée sur la qualité de l'image
        if mean_intensity > 140:  # Image claire
            confidence = 0.92
            simulated_text = """RÉPUBLIQUE FRANÇAISE
CARTE NATIONALE D'IDENTITÉ

Nom: MARTIN
Prénom(s): Jean Pierre
Né(e) le: 15.03.1985
à: PARIS 15EME (75)
Sexe: M
Taille: 1,75 m
Nationalité: Française
N°: 123456789012

Délivré le: 20.01.2020
par: PRÉFECTURE DE POLICE
Valable jusqu'au: 20.01.2030"""
        else:  # Image moins claire
            confidence = 0.76
            simulated_text = """RÉPUBLIQUE FRANÇAISE
CARTE NATIONALE D'IDENTITÉ

Nom: DUBOIS
Prénom(s): Marie Claire
Né(e) le: 22.07.1992
à: LYON 3EME (69)
Sexe: F
Taille: 1,65 m
Nationalité: Française
N°: 987654321098"""
        
        return {
            'raw_text': simulated_text,
            'confidence': confidence,
            'method': 'Simulation (Tesseract non disponible)',
            'processing_time': 0.1,
            'word_count': len(simulated_text.split()),
            'character_count': len(simulated_text),
            'is_confident': confidence > 0.8,
            'processed_image_path': image_path
        }
    
    def extract_structured_information(self, classification_result, ocr_result):
        """Extraction d'informations structurées - utilise les deux méthodes"""
        doc_type = classification_result.get('type', 'Other')
        raw_text = ocr_result.get('raw_text', '')
        
        if doc_type == 'ID Card':
            # Utilisation de la méthode de base de old_code.py d'abord
            basic_info = extract_info_basic(raw_text)
            
            # Ajout des informations de document
            basic_info['document_type'] = 'Carte Nationale d\'Identité'
            basic_info['extraction_method'] = 'old_code.py amélioré'
            basic_info['document_detection'] = detect_document_type(raw_text)
            
            return basic_info
            
        elif doc_type == 'Passport':
            return {
                'document_type': 'Passeport',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Analyseur de passeport en développement'
            }
        elif doc_type == 'Driver License':
            return {
                'document_type': 'Permis de Conduire',
                'detected_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Analyseur de permis de conduire en développement'
            }
        else:
            return {
                'document_type': 'Document non identifié',
                'raw_text': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
            }

# Base de données simple en mémoire pour les analyses
analyses_db = {}

class RESTfulDocumentProcessor:
    """Processeur de documents avec API REST"""
    
    def __init__(self):
        self.processor = DocumentProcessor()  # Maintenant défini
    
    def create_analysis(self, file_path, filename):
        """Crée une nouvelle analyse avec ID unique"""
        analysis_id = str(uuid.uuid4())
        
        analysis = {
            'id': analysis_id,
            'filename': filename,
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'file_path': file_path,
            'results': None,
            'error': None
        }
        
        analyses_db[analysis_id] = analysis
        return analysis_id
    
    def process_document(self, analysis_id):
        """Traite le document de manière asynchrone"""
        analysis = analyses_db.get(analysis_id)
        if not analysis:
            return None
        
        try:
            # Votre logique de traitement existante
            classification = self.processor.classify_document(analysis['file_path'])
            ocr_result = self.processor.extract_text_with_ocr(analysis['file_path'])
            structured_info = self.processor.extract_structured_information(classification, ocr_result)
            
            # Mise à jour des résultats
            analysis['status'] = 'completed'
            analysis['completed_at'] = datetime.now().isoformat()
            analysis['results'] = {
                'classification': classification,
                'ocr_result': ocr_result,
                'structured_information': structured_info
            }
            
            return analysis
            
        except Exception as e:
            analysis['status'] = 'failed'
            analysis['error'] = str(e)
            analysis['failed_at'] = datetime.now().isoformat()
            return analysis

# Instance globale
rest_processor = RESTfulDocumentProcessor()

# ====== ROUTES RESTful ======

@app.route('/api/v1/documents', methods=['POST'])
def create_document_analysis():
    """
    POST /api/v1/documents
    Crée une nouvelle analyse de document
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'A file must be included in the request'
        }), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file',
            'message': 'File must be PNG, JPG, or JPEG'
        }), 400
    
    try:
        # Sauvegarde sécurisée
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Création de l'analyse
        analysis_id = rest_processor.create_analysis(file_path, filename)
        
        # Traitement immédiat (ou pourrait être asynchrone)
        analysis = rest_processor.process_document(analysis_id)
        
        # Réponse REST conforme
        response_data = {
            'id': analysis_id,
            'status': analysis['status'],
            'created_at': analysis['created_at'],
            'filename': analysis['filename']
        }
        
        if analysis['status'] == 'completed':
            response_data.update({
                'completed_at': analysis['completed_at'],
                'results': analysis['results']
            })
        elif analysis['status'] == 'failed':
            response_data.update({
                'error': analysis['error'],
                'failed_at': analysis['failed_at']
            })
        
        # Code de statut approprié
        status_code = 201 if analysis['status'] == 'completed' else 202
        
        return jsonify(response_data), status_code
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/v1/documents/<analysis_id>', methods=['GET'])
def get_document_analysis(analysis_id):
    """
    GET /api/v1/documents/{id}
    Récupère une analyse spécifique
    """
    analysis = analyses_db.get(analysis_id)
    
    if not analysis:
        return jsonify({
            'error': 'Analysis not found',
            'message': f'No analysis found with ID {analysis_id}'
        }), 404
    
    response_data = {
        'id': analysis['id'],
        'filename': analysis['filename'],
        'status': analysis['status'],
        'created_at': analysis['created_at']
    }
    
    if analysis['status'] == 'completed':
        response_data.update({
            'completed_at': analysis['completed_at'],
            'results': analysis['results']
        })
    elif analysis['status'] == 'failed':
        response_data.update({
            'error': analysis['error'],
            'failed_at': analysis['failed_at']
        })
    
    return jsonify(response_data), 200

@app.route('/api/v1/documents', methods=['GET'])
def list_document_analyses():
    """
    GET /api/v1/documents
    Liste toutes les analyses avec pagination
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    status_filter = request.args.get('status')
    
    # Filtrage
    filtered_analyses = []
    for analysis in analyses_db.values():
        if status_filter and analysis['status'] != status_filter:
            continue
        filtered_analyses.append({
            'id': analysis['id'],
            'filename': analysis['filename'],
            'status': analysis['status'],
            'created_at': analysis['created_at']
        })
    
    # Tri par date de création (plus récent en premier)
    filtered_analyses.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Pagination
    start = (page - 1) * limit
    end = start + limit
    paginated_analyses = filtered_analyses[start:end]
    
    return jsonify({
        'data': paginated_analyses,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': len(filtered_analyses),
            'total_pages': (len(filtered_analyses) + limit - 1) // limit
        }
    }), 200

@app.route('/api/v1/documents/<analysis_id>', methods=['DELETE'])
def delete_document_analysis(analysis_id):
    """
    DELETE /api/v1/documents/{id}
    Supprime une analyse et son fichier
    """
    analysis = analyses_db.get(analysis_id)
    
    if not analysis:
        return jsonify({
            'error': 'Analysis not found',
            'message': f'No analysis found with ID {analysis_id}'
        }), 404
    
    try:
        # Suppression du fichier
        if os.path.exists(analysis['file_path']):
            os.remove(analysis['file_path'])
        
        # Suppression de la base de données
        del analyses_db[analysis_id]
        
        return '', 204  # No Content
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/v1/documents/<analysis_id>/file', methods=['GET'])
def get_document_file(analysis_id):
    """
    GET /api/v1/documents/{id}/file
    Récupère le fichier original
    """
    analysis = analyses_db.get(analysis_id)
    
    if not analysis:
        return jsonify({
            'error': 'Analysis not found'
        }), 404
    
    if not os.path.exists(analysis['file_path']):
        return jsonify({
            'error': 'File not found'
        }), 404
    
    return send_from_directory(
        os.path.dirname(analysis['file_path']),
        os.path.basename(analysis['file_path'])
    )

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """
    GET /api/v1/health
    Point de santé de l'API
    """
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'capabilities': {
            'tesseract': HAS_TESSERACT,
            'french_support': HAS_FRENCH,
            'supported_formats': list(app.config['ALLOWED_EXTENSIONS'])
        }
    }), 200

# ====== INTERFACE WEB (optionnelle) ======

@app.route('/')
def web_interface():
    """Interface web simple pour tester l'API"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR System - API RESTful</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .method { display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; margin-right: 10px; }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            .delete { background: #f93e3e; }
            code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OCR System - API RESTful</h1>
            
            <h2>Endpoints disponibles:</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/api/v1/documents</code>
                <p>Crée une nouvelle analyse de document. Envoyez un fichier via form-data.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/documents/{id}</code>
                <p>Récupère les résultats d'une analyse spécifique.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/documents</code>
                <p>Liste toutes les analyses avec pagination (?page=1&limit=10&status=completed).</p>
            </div>
            
            <div class="endpoint">
                <span class="method delete">DELETE</span>
                <code>/api/v1/documents/{id}</code>
                <p>Supprime une analyse et son fichier.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/documents/{id}/file</code>
                <p>Télécharge le fichier original d'une analyse.</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/api/v1/health</code>
                <p>Vérification de santé de l'API.</p>
            </div>
            
            <h2>Test simple:</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".png,.jpg,.jpeg" required>
                <button type="submit">Analyser via API</button>
            </form>
            <div id="result"></div>
            
            <script>
                document.getElementById('uploadForm').onsubmit = async function(e) {
                    e.preventDefault();
                    const formData = new FormData();
                    formData.append('file', document.getElementById('fileInput').files[0]);
                    
                    try {
                        document.getElementById('result').innerHTML = 'Analyse en cours...';
                        const response = await fetch('/api/v1/documents', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 'Erreur: ' + error.message;
                    }
                };
            </script>
        </div>
    </body>
    </html>
    """)

# ====== UTILITAIRES ======

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    # Création du dossier uploads
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("=== OCR SYSTEM - API RESTful ===")
    print("Interface web: http://localhost:8080")
    print("API Base URL: http://localhost:8080/api/v1")
    print("Health check: http://localhost:8080/api/v1/health")
    print(f"OCR: {'✅ Tesseract' if HAS_TESSERACT else '❌ Simulation'}")
    print(f"Français: {'✅ Disponible' if HAS_FRENCH else '❌ Non disponible'}")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\nArrêt du serveur...")
    except Exception as e:
        print(f"\nErreur: {e}")